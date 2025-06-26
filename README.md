# Megastructure Assembly via Collaborative Robots in Orbits (MACRO)

This repository contains simulation code accompanying the paper *Scalable Techniques for Autonomous Construction of a Paraboloidal Space Telescope in an Elliptic Orbit* by Aaron John Sabu and Dwaipayan Mukherjee.

## Overview

MACRO provides tools to simulate autonomous formation flying of multiple spacecraft for the construction of a large parabolic mirror while in orbit. The code implements formation maneuvers, attitude consensus and auction based allocation strategies used in the publication. Generated plots and animations from sample runs are available under the `results/` directory.

<!-- ## Abstract

It is well acknowledged that human-made technology is not always at par with human curiosity, and an example is the inability to send large telescopes to outer space despite their higher resolution and less atmospheric interference. In this paper, we develop a framework for autonomous in-orbit construction using spacecraft formation such that a large telescope can be built in an elliptic orbit using multiple spacecraft. We split this problem into four steps for converging the position and attitude of each spacecraft at predefined values around a central spacecraft. Each spacecraft performs attitude synchronization with its neighbors to match its three degrees of freedom in orientation as a parabolic mirror. Simulations validate our proposed methods and the paper concludes with an open possibility of using other techniques to improve upon existing results. -->

## Installation

This project requires Python 3.11. To set up the development environment using Conda:

```bash
# Create a new environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate macro

# (Optional) Install the project in editable mode with dev dependencies
pip install -e .[dev]
```

## Usage

Run the main simulation entry point:

```bash
python run.py
```

Simulation output will be written to HTML files in the working directory. 
Adjust parameters in `src/main.py` to explore different scenarios.

## Repository Structure

```
.
├── config/               # YAML configuration files for simulations
│   └── default.yaml
├── results/              # Output directory for plots, logs, and animations
├── src/
│   └── macro/            # Core simulation and control modules
│       ├── __init__.py
│       ├── auction.py
│       ├── benchmark.py
│       ├── glideslope.py
│       ├── ifmea.py
│       ├── main.py       # Contains the Macro class (main simulation)
│       ├── maneuvers.py
│       ├── mirror.py
│       ├── neighbors.py
│       ├── simulate.py
│       └── utils.py
├── tests/                # Unit tests
│   └── test_main.py
├── run.py                # Top-level script to run a simulation
├── environment.yml       # Conda environment definition (Python 3.11)
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # Optional pip-based install support
├── README.md             # Project overview and setup instructions
├── LICENSE               # License file
└── .gitignore
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and development instructions.

## Citation

If you use this code in your research, please cite our ICC 2021 paper:

```bibtex
@article{sabu2021scalable,
  author    = {John-Sabu, Aaron and Mukherjee, Dwaipayan},
  title     = {Scalable Techniques for Autonomous Construction of a Paraboloidal Space Telescope in an Elliptic Orbit},
  booktitle = {2021 Seventh Indian Control Conference (ICC)},
  year      = {2021},
  pages     = {329-334},
  doi       = {10.1109/ICC54714.2021.9703189},
  url       = {https://ieeexplore.ieee.org/document/9703189}
}
```

A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

This project is licensed under the terms of the GNU General Public License version 2. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported in part by an ISRO-funded project bearing code RD/0120-ISROC00-007. Aaron John Sabu was an undergraduate student at the time of writing and Dwaipayan Mukherjee was an Assistant Professor in the Department of Electrical Engineering, IIT Bombay.