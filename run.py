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

import argparse
from src.macro.main import main

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--config", type=str, default="config/default.yaml", help="Path to YAML config"
  )
  args = parser.parse_args()

  main(args.config)
