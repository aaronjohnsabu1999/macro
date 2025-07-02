# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Indian Institute of Technology, Bombay                  *
# *                                                         *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# *                                                         *
# ***********************************************************/

import logging
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment

from macro.utils import gen_cost, NullLogger


auction_map = {
    "Standard": "StandardAuction",
    "Hybrid": "HybridAuction",
    "Distributed": "DistributedAuction",
    "Consensus": "ConsensusAuction",
    "Greedy": "GreedyAuction",
}


class Auction(ABC):
    """
    Abstract base class for auction-based task allocation algorithms.

    Attributes:
        R_0 (np.ndarray): Initial positions of agents.
        R_f (np.ndarray): Target positions to assign.
        G (np.ndarray): Adjacency matrix indicating communication graph.
        num_agents (int): Number of agents.
    """

    def __init__(self, R_0, R_f, G, *args, **kwargs):
        self.R_0 = R_0
        self.R_f = R_f
        self.G = G
        self.num_agents = len(R_0)
        self.logger = kwargs.get("logger", NullLogger())
        self.logger.debug("Auction initialized with %d agents", self.num_agents)

    @abstractmethod
    def assign(self):
        """
        Abstract method to compute a task assignment.
        Must be implemented by subclasses.

        Returns:
            np.ndarray: New target positions after assignment.
        """
        pass


class StandardAuction(Auction):
    """
    Implements the standard distributed auction using local Hungarian assignments.
    """

    def assign(self):
        """
        Assigns tasks using the Hungarian algorithm for each agent's neighborhood.

        Returns:
            np.ndarray: Updated target assignments.
        """
        self.logger.debug("Starting standard auction assignment")
        for ego in range(self.num_agents):
            neighbors = [i for i in range(self.num_agents) if self.G[ego][i] > 0]
            R_fn = [self.R_f[i] for i in neighbors]
            R_0n = [self.R_0[i] for i in neighbors]
            cost_matrix = [gen_cost(r_0, R_fn) for r_0 in R_0n]
            _, col_ind = linear_sum_assignment(cost_matrix)
            for idx, agent in enumerate(neighbors):
                self.R_f[agent] = R_fn[col_ind[idx]]
        return self.R_f


class GreedyAuction(Auction):
    """
    Implements a greedy resolution of path conflicts by locally swapping assignments.
    """

    def _check_intersections(self, ego):
        """
        Checks for 3D line intersection between ego's path and its neighbors'.

        Args:
            ego (int): Index of the ego agent.

        Returns:
            tuple[bool, list[int]]: Whether intersection exists, and list of colliding agents.
        """
        self.logger.debug("Checking intersections for agent %d", ego)
        r_0, r_f = self.R_0[ego], self.R_f[ego]
        intersecting = []
        for i, (r_0n, r_fn) in enumerate(zip(self.R_0, self.R_f)):
            if self.G[ego][i] and i != ego:
                A = [
                    [r_0[0] - r_f[0], -(r_0n[0] - r_fn[0])],
                    [r_0[1] - r_f[1], -(r_0n[1] - r_fn[1])],
                ]
                B = [r_f[0] - r_fn[0], r_f[1] - r_fn[1]]
                try:
                    t, s = np.linalg.solve(A, B)
                    dz = abs(
                        (r_0[2] - r_f[2]) * t
                        - (r_0n[2] - r_fn[2]) * s
                        - (r_f[2] - r_fn[2])
                    )
                    if dz < 1e-3:
                        intersecting.append(i)
                except np.linalg.LinAlgError:
                    continue
        return bool(intersecting), intersecting

    def assign(self):
        """
        Iteratively swaps assignments to resolve intersections.

        Returns:
            np.ndarray: Updated target assignments.
        """
        self.logger.debug("Starting greedy auction assignment")
        while True:
            changes = False
            for ego in range(self.num_agents):
                intersects, neighbors = self._check_intersections(ego)
                if intersects:
                    for other in neighbors:
                        self.R_f[ego], self.R_f[other] = self.R_f[other], self.R_f[ego]
                        if not self._check_intersections(ego)[0]:
                            changes = True
                            break
            if not changes:
                break
        return self.R_f


class DistributedAuction(Auction):
    """
    Implements the distributed auction algorithm by Zavlanos et al.
    """

    def assign(self):
        """
        Runs distributed auction with local bidding and price updates.

        Returns:
            np.ndarray: Final assigned targets after convergence.
        """
        self.logger.debug("Starting distributed auction assignment")
        if not self.G.is_connected():
            return self.R_f

        beta = [gen_cost(r_0, self.R_f, False) for r_0 in self.R_0]
        a = np.zeros(self.num_agents, dtype=int)
        p = np.zeros((self.num_agents, self.num_agents))
        b = np.zeros_like(p)

        while len(a) != len(set(a)):
            a_next = np.copy(a)
            for ego in range(self.num_agents):
                p_max = np.max(
                    [p[k] for k in range(self.num_agents) if self.G[ego][k]], axis=0
                )
                b_max = np.max(
                    [
                        b[k] * (p[k] == p_max)
                        for k in range(self.num_agents)
                        if self.G[ego][k]
                    ],
                    axis=0,
                )

                if p[ego][a[ego]] <= p_max[a[ego]] and b_max[a[ego]] != ego:
                    scores = [beta[ego][j] - p_max[j] for j in range(self.num_agents)]
                    a_next[ego] = np.argmax(scores)
                    b[ego][a_next[ego]] = ego
                    v = max(scores)
                    w = max(score for j, score in enumerate(scores) if j != a_next[ego])
                    gamma = v - w + np.random.rand() * 0.05
                    p[ego][a_next[ego]] += gamma
                else:
                    a_next[ego] = a[ego]
            if np.array_equal(a, a_next):
                break
            a = a_next
        return [self.R_f[i] for i in a]


# 'Consensus-Based Decentralized Auctions for Robust Task Allocation' - Choi, Brunet, How
class ConsensusAuction(Auction):
    """
    Implements the consensus-based decentralized auction by Choi, Brunet, and How.
    """

    def assign(self):
        """
        Uses consensus mechanism to agree on unique task assignments.

        Returns:
            np.ndarray: Final target assignments after consensus.
        """
        self.logger.debug("Starting consensus auction assignment")
        if not self.G.is_connected():
            return self.R_f

        c = [gen_cost(r_0, self.R_f) for r_0 in self.R_0]
        x, y, z = (
            np.zeros((self.num_agents, self.num_agents)),
            np.zeros_like(x),
            np.zeros_like(x),
        )
        J = [0] * self.num_agents

        def select_task(ego):
            """
            Internal subroutine for ego agent to select the next best task.
            """
            if np.sum(x[ego]) == 0:
                h = c[ego] >= y[ego]
                if np.any(h):
                    J[ego] = np.argmax(h * c[ego])
                    x[ego][J[ego]] = 1
                    y[ego][J[ego]] = c[ego][J[ego]]

        def update_task(ego):
            """
            Internal subroutine to update ego's perception of task ownership.
            """
            for j in range(self.num_agents):
                y[ego][j] = max(
                    y[k][j] for k in range(self.num_agents) if self.G[ego][k]
                )
            z[ego][J[ego]] = np.argmax(
                [y[k][J[ego]] for k in range(self.num_agents) if self.G[ego][k]]
            )
            if z[ego][J[ego]] != ego:
                x[ego][J[ego]] = 0

        while not all(np.any(xi) for xi in x):
            for ego in range(self.num_agents):
                select_task(ego)
            for ego in range(self.num_agents):
                update_task(ego)
        return self.R_f


class HybridAuction(Auction):
    """
    Combines greedy and distributed auction based on connectivity.

    If the graph is not connected, falls back to greedy conflict resolution.
    """

    def assign(self):
        """
        Choose between greedy and distributed auction depending on connectivity.

        Returns:
            np.ndarray: Final target assignments.
        """
        self.logger.debug("Starting hybrid auction assignment")
        if not self.G.is_connected():
            self.logger.debug("Graph is not connected, using greedy auction")
            return GreedyAuction(self.R_0, self.R_f, self.G).assign()
        else:
            self.logger.debug("Graph is connected, using distributed auction")
            return DistributedAuction(self.R_0, self.R_f, self.G).assign()


if __name__ == "__main__":
    R_0 = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])  # Initial positions
    R_f = np.array([[3, 3, 0], [4, 4, 0], [5, 5, 0]])  # Target positions
    G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Connectivity graph

    logger = logging.getLogger("Auction")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    auction = HybridAuction(R_0, R_f, G=G, logger=logger)
    new_targets = auction.assign()
    print("New Assignments:", new_targets)
