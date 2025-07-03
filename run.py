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
import argparse
import numpy as np
from macro import Macro

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger("macro")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Seed the random number generator for reproducibility
    np.random.seed(42)
    # Initialize and run the Macro simulation
    sim = Macro(config_path=args.config, logger=logger)
    sim.run()
