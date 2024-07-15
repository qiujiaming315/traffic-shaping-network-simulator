import numpy as np
import os

from lib.network_simulator import NetworkSimulator
from viz.packet_delay_demo import plot_delay_distribution


if __name__ == '__main__':
    """An example use case of the traffic shaping network simulator."""
    # Specify the path to the input data (flow profile + flow routes) of the simulation. Replace with your own path.
    flow_path = f"./data/flow/flow.npy"
    route_path = f"./data/route/route.npy"
    # Load the input data.
    flow_profile = np.load(flow_path)
    flow_route = np.load(route_path)
    # Assuming full shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    # Set the random seed for reproducible results.
    np.random.seed(0)
    # Specify the output directory to save the simulation results. Replace with your own choice of directory path.
    output_path = "./figures/"
    os.makedirs(output_path, exist_ok=True)
    # Establish the simulator. Make your own choice of the input parameters.
    simulator = NetworkSimulator(flow_profile, flow_route, shaping_delay, simulation_time=100.0,
                                 shaping_mode="per_flow", arrival_pattern_type="sync_burst", awake_dur=10.0,
                                 arrival_pattern=None, scaling_factor=1.0, packet_size=1)
    # Start the simulation.
    simulator.simulate()
    # Check the statistics of interest (e.g., packet arrival time, departure time at each hop, and the desired
    # latency target) after the simulation.
    arrival_time = simulator.arrival_time
    departure_time = simulator.departure_time
    latency_target = simulator.latency_target
    # You can retrieve other statistics, such as the (normalized) end-to-end delay of each packet.
    delay_aggregate = []
    for flow_arrival, flow_departure, flow_delay_target in zip(arrival_time, departure_time, latency_target):
        for arrival, departure in zip(flow_arrival, flow_departure):
            delay_aggregate.append([(departure[-1] - arrival) / flow_delay_target * 100])
    # You can finally use the built-in plotting functionalities to visualize the data and save the results.
    # For example, you may plot the distribution of the (normalized) end-to-end delay.
    plot_delay_distribution(delay_aggregate, output_path, "per_flow_shaping")
