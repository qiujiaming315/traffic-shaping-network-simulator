# Discrete Event Simulator for Traffic Shaping Network

## Overview

This repository provides implementation of a discrete event simulator for traffic shaping networks in Python. The simulator covers several schedulers, shapers, and shaping strategies.

### Scheduler

We currently support First-In-First-Out (FIFO) scheduler and will extend to static priority and deadline-based schedulers in the future.

### Traffic Shaper

We provide traffic shapers in the form of packetized token buckets. Particularly, a `MultiSlopeShaper` module implements a combination of multiple token buckets, i.e., a packet is released by the module when released by every token bucket in the collection.

<!-- TODO: Explain the shaping policy (fully reprofiling). -->

### Shaping Strategy

We provide shaping strategies with different levels of complexity, including per user flow shaping at every hop, interleaved shaping at every hop, shaping only at network ingress, and no shaping.

## Requirements

We recommend a recent Python 3.7+ distribution of [Anaconda](https://www.anaconda.com/products/individual).

To keep a local copy of our code, clone this repository and navigate into the cloned directory. You should then be ready to run the scripts once you installed all the pre-requisite packages.

```
# First navigate to the directory where you want to keep a copy of the code.
git clone https://github.com/qiujiaming315/traffic-shaping-network-simulator.git
# Navigate into the cloned directory to get started.
cd traffic-shaping-network-simulator
# Install all the required modules.
pip install -r requirements.txt
```

## Usage

### Generating Network Topology and Flow Profile

<!-- Our algorithm takes network topology and flow profile as inputs. Before running the algorithm, you need to first generate some input data.

We provide scripts that facilitate generating network profile, flow profile, and link weights available through `create_network.py`, `create_flow.py`, and `create_weight.py` respectively in the `input/` sub-directory. Each script allows you to either specify your own profile or generate profile using the build-in functions. You can try these different options by modifying the `__main__` function (with example code snippets for each option) of each script.

Once you modified the `__main__` function according to the desired configurations, you can directly run those scripts through command lines.

```
# Navigate into the input/ sub-directory.
cd input
# Generate network profile, flow profile, and (optional) link weights.
python create_network.py
python create_flow.py
python create_weight.py
``` -->

#### Network Profile

<!-- We use network profile to specify network topology as well as the route of each flow in the network.  The network profile is represented as an `m × n` matrix, with `m` and `n` being the number of traffic flows and network nodes respectively.

The matrix may either be of type `bool` for feed-forward network or `int` for cyclic network. The following figure demonstrates an example of retrieving network profile matrices given the graph representations of the networks.

<p align="center">
    <img src="img/network_profile.png" alt="network_profile" width="80%"/>
</p> -->

#### Flow Profile

<!-- We use flow profile to specify the token bucket parameters (rate, burst size) as well as the end-to-end latency target of each flow in the network. The flow profile is represented as an `m × 3` matrix, where `m` is the number of flows. The three columns stand for rate, burst size, and latency target respectively. -->

### Running the Simulation

<!-- The main script for running the minimization algorithm is `optimization.py`. Running it with `-h` gives you a detailed description over a list of parameters you can control:

- `net`: path to the input network profile.
- `flow`: path to the input flow profile.
- `out`: directory to save the output file.
- `file_name`: name of the file to save results.
- `--scheduler`: type of scheduler applied to each hop of the network. 0 for FIFO and 1 for SCED.
- `--objective`: type of the objective function to minimize. Available choices include: 0 for the sum of link bandwidth, 1 for weighted sum of link bandwidth, 2 for maximum link bandwidth.
- `--weight`: path to the link weights if the objective function is selected to be a weighted sum of link bandwidth.
- `--mode`: bandwidth minimization algorithm to run, 0 for NLP-based algorithm, 1 for the greedy algorithm. Greedy algorithm is applied by default.

For example, to compute the minimum required sum of link bandwidth with SCED schedulers using the greedy algorithm with network profile saved in `input/network/3/net1.npy` and flow profile saved in `input/flow/3/flow1.npz`, and save the results to `output/` under the name `result.npz`, you should use

```
# Make sure you are in the root directory of this repo,
# where optimization.py is stored.
python optimization.py input/network/3/net1.npy input/flow/3/flow1.npz output result --scheduler 1 --objective 0 --mode 1
``` -->

### Library

<!-- We factor various parts of the code into different modules in the `lib/`
directory. You can begin by looking at the main optimization script to see how
to make use of these modules.

- `utils.py`: Implement various utility functions (*e.g.,* function to load input data).
- `network_parser.py`: Helper functions that faciliate parsing the inputs and the solutions of the optimization.
- `genetic.py`: A parent class that provides a generic implementation of the genetic algorithm.
- `genetic_fifo.py`: Implement a genetic algorithm that performs guided random search to find the best solution based on solving multiple non-linear programs, for networks with FIFO schedulers.
- `genetic_sced.py`: Similar to `genetic_fifo.py`, but for networks with SCED schedulers.
- `heuristic_fifo.py`: Implement Greedy and the baseline solutions for networks with FIFO schedulers.
- `heuristic_sced.py`: Similar to `heuristic_fifo.py`, but for networks with SCED schedulers.
- `order_generator.py`: Implement various functions that handle flow orderings.
- `octeract.py`: Formulate the minimization problem into NLPs and call the Octeract engine to solve the generated NLPs. -->

## License

This code is being released under the [MIT License](LICENSE).
