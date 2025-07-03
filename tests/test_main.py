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

import os
import numpy as np
import pytest
from macro import Macro

CONFIG_PATH = "config/default.yaml"

def test_macro_run_success(tmp_path):
    """Test that the full simulation pipeline runs without crashing and outputs expected results."""

    sim = Macro(config_path=CONFIG_PATH)
    sim.run()

    # Basic shape assertions
    assert sim.trajectory.shape[0] == sim.num_agents
    assert sim.trajectory.shape[2] == 6  # (x, y, z, vx, vy, vz)
    assert sim.attitudes.shape[0] == sim.num_agents
    assert sim.attitudes.shape[2] == 3  # (roll, pitch, yaw)

    # Check that all stages were recorded
    for key in ["1_rendezvous", "2_formation", "3_ifmea"]:
        assert key in sim.stage_changes
        assert sim.stage_changes[key] is not None

def test_neighbors_consistency():
    """Ensure neighbor graph includes all agents after IFMEA."""
    sim = Macro(config_path=CONFIG_PATH)
    sim.run()

    neighbor_graph = sim.all_neighbors[-1]
    missing_agents = set(range(sim.num_agents)) - set(neighbor_graph.keys())

    # All agents should have entries (even if some have no neighbors)
    assert len(missing_agents) == 0, f"Missing neighbors for agents: {missing_agents}"

def test_attitude_stability():
    """Ensure attitude values are finite and bounded."""
    sim = Macro(config_path=CONFIG_PATH)
    sim.run()

    assert np.all(np.isfinite(sim.attitudes)), "Attitude matrix contains NaNs or Infs"
    max_attitude_change = np.max(np.abs(np.diff(sim.attitudes, axis=1)))
    assert max_attitude_change < 1e-1, "Attitude values vary too drastically"