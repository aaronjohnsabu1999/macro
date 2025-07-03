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

import copy
import logging
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment

from macro.graph import SystemGraph
from macro.utils import generate_cost, NullLogger, check_intersection


class Auction(ABC):
    """
    Abstract base class for auction-based task allocation algorithms.

    Attributes:
        initial_positions (np.ndarray): Initial positions of agents.
        target_positions (np.ndarray): Target positions to assign.
        graph (np.ndarray): Adjacency matrix indicating communication graph.
        num_agents (int): Number of agents.
    """

    def __init__(
        self,
        initial_positions: np.ndarray,
        target_positions: np.ndarray,
        graph: SystemGraph,
        *args,
        **kwargs,
    ):
        self.initial_positions = initial_positions
        self.target_positions = target_positions
        self.graph = graph
        self.num_agents = initial_positions.shape[0]
        self.logger = kwargs.get("logger", NullLogger())

    @abstractmethod
    def assign(self):
        """
        Abstract method to compute a task assignment.
        Must be implemented by subclasses.

        Returns:
            np.ndarray: New target positions after assignment.
        """
        return self.target_positions


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
        for ego in range(self.num_agents):
            neighbors = self.graph.get_neighbors(ego)
            if not neighbors:
                continue
            neighbors_initial_positions = self.initial_positions[neighbors]
            neighbors_target_positions = self.target_positions[neighbors]
            cost_matrix = np.stack(
                [
                    generate_cost(pos, neighbors_target_positions)
                    for pos in neighbors_initial_positions
                ]
            )
            if cost_matrix.size == 0:
                self.logger.debug(f"No neighbors for agent {ego}, skipping assignment")
                continue
            self.logger.debug(
                f"Agent {ego} neighbors: {neighbors}, cost matrix:\n{cost_matrix}"
            )
            row, col = linear_sum_assignment(cost_matrix)
            for c_idx, neighbor in zip(col, self.graph.get_neighbors(ego)):
                self.target_positions[neighbor] = neighbors_target_positions[c_idx]
        return self.target_positions


class GreedyAuction(Auction):
    """
    Implements a greedy resolution of path conflicts by locally swapping assignments.
    """

    def assign(self):
        """
        Iteratively swaps assignments to resolve intersections.

        Returns:
            np.ndarray: Updated target assignments.
        """
        while True:
            swaps = []
            for ego in range(self.num_agents):
                for other in self.graph.get_neighbors(ego):
                    if ego < other and check_intersection(
                        self.initial_positions[ego],
                        self.target_positions[ego],
                        self.initial_positions[other],
                        self.target_positions[other],
                    ):
                        swaps.append((ego, other))
            if not swaps:
                self.logger.debug("No intersections found, assignment complete")
                break
            for ego, other in swaps:
                self.target_positions[ego, other] = self.target_positions[other, ego]
                self.initial_positions[ego, other] = self.initial_positions[other, ego]

        return self.target_positions


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
        if not self.graph.is_connected():
            return self.target_positions

        utility_matrix = np.stack(
            [
                generate_cost(pos, self.target_positions, False)
                for pos in self.initial_positions
            ]
        )  # Utility matrix: utility_matrix[agent, task]
        curr_assignment = np.arange(
            self.num_agents, dtype=int
        )  # Each agent's current task choice
        prices = np.zeros(
            (self.num_agents, self.num_agents)
        )  # Price matrix: price_matrix[agent, task]
        bids = np.zeros_like(
            prices
        )  # Bids matrix: bids[agent, task] = agent ID if agent bids on task

        while True:
            next_assignment = curr_assignment.copy()
            for ego in range(self.num_agents):
                neighbors = self.graph.get_neighbors(ego)
                if not neighbors:
                    continue
                # Highest known prices among neighbors
                max_price = np.max(prices[neighbors], axis=0)
                # Resolve ties: among neighbors whose price equals max, get their IDs
                neighbor_bids = bids[neighbors] * (prices[neighbors] == max_price)
                max_bids = np.max(neighbor_bids, axis=0)

                # If the neighbor bid invalidates our current assignment, consider rebidding
                if (
                    prices[ego, curr_assignment[ego]] <= max_price[curr_assignment[ego]]
                    and max_bids[curr_assignment[ego]] != ego
                ):
                    # Evaluate net utilities
                    net_utilities = utility_matrix[ego] - max_price
                    best_task = np.argmax(net_utilities)
                    next_assignment[ego] = best_task
                    bids[ego, best_task] = ego

                    # Price increment based on difference between first and second best utilities
                    highest_gain = net_utilities[best_task]
                    alt_utilities = np.delete(net_utilities, best_task)
                    second_gain = (
                        np.max(alt_utilities) if alt_utilities.size > 0 else 0.0
                    )
                    price_increment = (
                        highest_gain - second_gain + np.random.rand() * 0.05
                    )
                    prices[ego, best_task] += price_increment
            if np.array_equal(curr_assignment, next_assignment):
                break
            curr_assignment = next_assignment
        # Update target positions according to converged assignment
        self.target_positions = self.target_positions[curr_assignment]
        return self.target_positions


class ConsensusAuction(Auction):
    """
    Implements the single-assignment Consensus-Based Auction Algorithm (CBAA) from Choi, Brunet & How (2009), with a bidding phase followed by local consensus.
    """

    def assign(self):
        """
        Uses consensus mechanism to agree on unique task assignments.

        Returns:
            np.ndarray: Final target assignments after consensus.
        """
        if not self.graph.is_connected():
            return self.target_positions

        # Bid utility matrix: bid_utility[i][j] is cost for agent i to task j
        bid_utility = np.stack(
            [
                generate_cost(pos, self.target_positions)
                for pos in self.initial_positions
            ]
        )

        # bid_flag[i][j] = 1 if agent i currently holds best bid on task j
        # bid_price[i][j] = currently known highest bid price for task j by i
        # bid_winner[i][j] = agent ID believed by i to be winning task j
        bid_flag = np.zeros((self.num_agents, self.num_agents), dtype=int)
        bid_price = np.zeros((self.num_agents, self.num_agents), dtype=float)
        bid_winner = np.full((self.num_agents, self.num_agents), -1, dtype=int)

        # Single task assignment per agent
        assignment = np.full(self.num_agents, -1, dtype=int)

        def auction_phase(agent):
            # If agent already holds a bid, do nothing
            if assignment[agent] >= 0:
                return

            utilities = bid_utility[agent]
            known_prices = bid_price[agent]
            feasible = utilities > known_prices
            if not np.any(feasible):
                return

            best_task = np.argmax(np.where(feasible, utilities, -np.inf))
            assignment[agent] = best_task
            bid_flag[agent, best_task] = 1
            bid_price[agent, best_task] = utilities[best_task]
            bid_winner[agent, best_task] = agent

        def consensus_phase(agent):
            for task in range(self.num_agents):
                # share max known bid price among neighbors
                neighbor_prices = [
                    bid_price[n, task] for n in self.graph.get_neighbors(agent)
                ]
                if neighbor_prices:
                    bid_price[agent, task] = max(
                        bid_price[agent, task], max(neighbor_prices)
                    )
                    # update winner to agent with max price
                    winners = [
                        (n, bid_price[n, task]) for n in self.graph.get_neighbors(agent)
                    ]
                    winners.append((agent, bid_price[agent, task]))
                    bid_winner[agent, task] = max(winners, key=lambda x: x[1])[0]

                # if agent lost the task it thought it held, revoke
                if assignment[agent] == task and bid_winner[agent, task] != agent:
                    assignment[agent] = -1
                    bid_flag[agent, task] = 0

        # Main loop: alternate auction + consensus until stable
        while True:
            old_assignment = assignment.copy()
            for agent in range(self.num_agents):
                auction_phase(agent)
            for agent in range(self.num_agents):
                consensus_phase(agent)
            if np.array_equal(assignment, old_assignment):
                break

        # apply assignments
        for i, task in enumerate(assignment):
            if task >= 0:
                self.target_positions[i] = self.target_positions[task]

        return self.target_positions


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
        if not self.graph.is_connected():
            self.logger.debug("Graph is not connected, using greedy auction")
            return GreedyAuction(
                self.initial_positions, self.target_positions, self.graph
            ).assign()
        else:
            self.logger.debug("Graph is connected, using distributed auction")
            return DistributedAuction(
                self.initial_positions, self.target_positions, self.graph
            ).assign()


auction_map = {
    "Standard": StandardAuction,
    "Distributed": DistributedAuction,
    "Consensus": ConsensusAuction,
    "Greedy": GreedyAuction,
    "Hybrid": HybridAuction,
}


if __name__ == "__main__":
    logger = logging.getLogger("AuctionTest")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Define layered configuration
    layer_sizes = [6, 12, 24]
    agent_types = [i + 1 for i, size in enumerate(layer_sizes) for _ in range(size)]
    agent_ids = list(range(sum(layer_sizes)))
    np.random.seed(0)
    np.random.shuffle(agent_ids)

    # Assign shuffled IDs to layers
    layer_config = []
    idx = 0
    for size in layer_sizes:
        layer_config.append(agent_ids[idx : idx + size])
        idx += size

    flat_config = [agent for layer in layer_config for agent in layer]
    graph = SystemGraph(layer_config, flat_config)

    # Create mock initial and target positions
    initial_positions = np.random.rand(len(flat_config), 3) * 100
    target_positions = np.random.rand(len(flat_config), 3) * 100

    logger.info("Initial positions:\n%s", initial_positions)
    logger.info("Original target positions:\n%s", target_positions)
    # Run all auctions for demo
    for name, AuctionClass in auction_map.items():
        if name is not "Distributed":
            # Skip DistributedAuction for this demo
            continue
        logger.info(f"Running {name} auction")
        auction = AuctionClass(
            initial_positions.copy(), target_positions.copy(), graph, logger=logger
        )
        new_targets = auction.assign()
        logger.info(f"Modified target positions: {new_targets}")
