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

from math import floor
from macro.graph import SystemGraph

from macro.utils import to_layer_index, to_flat_index


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
    closest = floor(mismatch_idx / 2)
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
    # layer_type is either in kwargs or it is 0 for the root and increases by 1 for each layer aka the index of the layer
    # the idea of ifmea is to work from the outermost layer inward, so we start with the outermost layer
    # we try to match agent_type to layer_type by performing rotations and exchanges
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

    # Work from outermost layer inward
    outer_layer_idx = len(layer_config) - 1
    while outer_layer_idx > 0:
        print(f"Processing layer {outer_layer_idx}...")
        inner_layer_idx = 0

        while inner_layer_idx != outer_layer_idx:
            print(f"Checking layer {inner_layer_idx}...")
            try:
                # Find position of agent needing to move up
                pos_idx = graph.get_layer_config()[inner_layer_idx].index(
                    outer_layer_idx + 1
                )
                print(f"Found position {pos_idx} in layer {inner_layer_idx}.")
            except ValueError:
                inner_layer_idx -= 1
                print(f"No agent to move up in layer {inner_layer_idx + 1}.")
                continue

            # Find closest mismatch position in the layer above
            mismatch_idx, target_pos_idx = find_nearest_mismatch(
                graph.get_layer_config()[inner_layer_idx],
                graph.get_layer_config()[inner_layer_idx + 1],
                inner_layer_idx,
                pos_idx,
            )
            print(
                f"Found mismatch at index {mismatch_idx} in layer {inner_layer_idx + 1}."
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
                print(f"Rotated layer {inner_layer_idx} by {shift} positions.")

            # Perform exchange (depending on whether mismatch is left or right)
            if mismatch_idx < 2 * target_pos_idx:
                third_idx = target_pos_idx - 1
            else:
                third_idx = (target_pos_idx + 1) % layer_lengths[inner_layer_idx]

            graph.exchange_positions(
                to_flat_index(layer_lengths, inner_layer_idx, target_pos_idx),
                to_flat_index(layer_lengths, inner_layer_idx + 1, mismatch_idx),
                to_flat_index(layer_lengths, inner_layer_idx, third_idx),
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
            print(
                f"Exchanged positions {target_pos_idx}, {mismatch_idx}, and {third_idx} in layers {inner_layer_idx} and {inner_layer_idx + 1}."
            )
            print("Layer config before step:", graph.get_layer_config())
            input()

            # Allow upward re-checking
            if inner_layer_idx < outer_layer_idx - 1:
                inner_layer_idx += 1

        outer_layer_idx -= 1

    return commands


if __name__ == "__main__":
    init_layer_config = [
        [1, 1, 3, 1, 2, 2],
        [2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ]
    print(ifmea_commands(init_layer_config))
