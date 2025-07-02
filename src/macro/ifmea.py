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

import numpy as np

from macro.graph import SystemGraph
from macro.utils import to_flat_index


def find_nearest_mismatch(config, next_config, layer_idx, pos_idx):
    """Find the nearest mismatch in the next layer configuration."""
    n_L_next = len(next_config)
    for dist in range(int(n_L_next / 2) + 1):
        upper = (pos_idx * 2 + dist) % n_L_next
        lower = (pos_idx * 2 - dist) % n_L_next
        if next_config[upper] != layer_idx + 2:
            mismatch_idx = upper
            break
        elif next_config[lower] != layer_idx + 2:
            mismatch_idx = lower
            break
    closest = int(np.floor(mismatch_idx / 2))
    if closest < pos_idx:
        closest += mismatch_idx % 2
    return mismatch_idx, closest


def ifmea_commands(layer_config, *args, **kwargs):
    """
    Generate rotation and exchange commands to transform a layered configuration
    into a final, ordered structure using the IFMEA algorithm.

    Parameters
    ----------
    layer_config : list[list[int]]
        Nested list representing agents in each concentric layer.
    *args, **kwargs :
        Optional arguments, including:
        - graph: pre-initialized SystemGraph object (optional)

    Returns
    -------
    list[tuple]
        A list of commands representing the transformations:
        - ("R", start_idx, end_idx, direction, steps) for rotation
        - ("E", idx1, idx2, idx3, 1) for three-agent exchanges
    """
    commands = []
    if not layer_config:
        raise ValueError("Layer configuration cannot be empty.")
    if not isinstance(layer_config, list) or not all(
        isinstance(layer, list) for layer in layer_config
    ):
        raise TypeError("Layer configuration must be a list of lists.")
    layer_type = kwargs.get("layer_type", list(range(len(layer_config))))
    if not isinstance(layer_type, list):
        raise TypeError("Layer type must be a list of integers.")
    if len(layer_config) != len(layer_type):
        raise ValueError(
            "Layer configuration and layer type must have the same number of layers."
        )

    flat_config = [agent for layer in layer_config for agent in layer]
    graph = kwargs.get("graph", SystemGraph(layer_config, flat_config))
    layer_lengths = graph.get_layer_lengths()

    print("Initial Layer Configuration:", layer_config)

    # Work from outermost layer inward
    outer_layer_idx = len(layer_config) - 1
    while outer_layer_idx > 0:
        inner_layer_idx = 0

        while inner_layer_idx != outer_layer_idx:
            try:
                # Find position of agent needing to move up
                pos_idx = graph.get_layer_config()[inner_layer_idx].index(
                    outer_layer_idx + 1
                )
            except ValueError:
                if inner_layer_idx == 0:
                    break
                inner_layer_idx -= 1
                continue

            # Find closest mismatch position in the layer above
            mismatch_idx, target_pos_idx = find_nearest_mismatch(
                graph.get_layer_config()[inner_layer_idx],
                graph.get_layer_config()[inner_layer_idx + 1],
                inner_layer_idx,
                pos_idx,
            )

            # Rotate to align source and target
            if target_pos_idx != pos_idx:
                shift = target_pos_idx - pos_idx
                graph.rotate_layer(inner_layer_idx, shift)
                commands.append(
                    (
                        "R",
                        to_flat_index(layer_lengths, inner_layer_idx, 0),
                        to_flat_index(layer_lengths, inner_layer_idx + 1, 0) - 1,
                        int(shift / abs(shift)),
                        abs(shift),
                    )
                )

            # Perform exchange (depending on whether mismatch is left or right)
            if mismatch_idx < 2 * target_pos_idx:
                third_idx = target_pos_idx - 1
            else:
                third_idx = (target_pos_idx + 1) % layer_lengths[inner_layer_idx]

            graph.exchange_positions(
                (inner_layer_idx, target_pos_idx),
                (inner_layer_idx + 1, mismatch_idx),
                (inner_layer_idx, third_idx),
            )
            commands.append(
                (
                    "E",
                    to_flat_index(layer_lengths, inner_layer_idx, target_pos_idx),
                    to_flat_index(layer_lengths, inner_layer_idx + 1, mismatch_idx),
                    to_flat_index(layer_lengths, inner_layer_idx, third_idx),
                    1,
                )
            )

            # Allow upward re-checking
            if inner_layer_idx < outer_layer_idx - 1:
                inner_layer_idx += 1

        outer_layer_idx -= 1

    print("Final Layer Configuration:", graph.get_layer_config())
    return commands


if __name__ == "__main__":
    # Define how many of each type we need
    counts = {
        1: 6,
        2: 12,
        3: 18,
        4: 24,
        5: 30,
    }

    # Create the full list of agents
    all_agents = [agent_type for agent_type, count in counts.items() for _ in range(count)]
    np.random.shuffle(all_agents)

    # Define layer sizes
    layer_sizes = [counts[i] for i in range(1, 6)]

    # Split shuffled agents into layers
    init_layer_config = []
    idx = 0
    for size in layer_sizes:
        init_layer_config.append(all_agents[idx:idx + size])
        idx += size

    print(ifmea_commands(init_layer_config))

