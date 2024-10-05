import argparse
import numpy as np
import os
import pickle
import time

from lib.network_simulator import NetworkSimulator
from viz.packet_delay_demo import plot_delay_distribution


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('flow_path', help="Path to the input npy files describing flow profiles.")
    args.add_argument('route_path', help="Path to the input npy files describing flow routes.")
    args.add_argument('out_dir', help="Directory to save results.")
    args.add_argument('file_name', help="Name fo the file to save results.")
    args.add_argument('--simulation-time', type=float, default=100.0,
                      help="Total simulation time (in seconds).")
    args.add_argument('--scheduling-policy', type=str, default="fifo",
                      help="Type of scheduler applied at each hop of the network. Choose between 'fifo' and 'sced'.")
    args.add_argument('--shaping-mode', type=str, default="per_flow",
                      help="Type of traffic shapers applied. Choose among 'per_flow', 'interleaved', 'ingress',"
                           "and 'none'. Only active when scheduling policy is 'fifo'")
    args.add_argument('--buffer-bound', type=str, default='infinite',
                      help="Link buffer bound. Choose between 'infinite', and 'with_shaping'.")
    args.add_argument('--arrival-pattern-type', type=str, default="sync_burst",
                      help="Type of traffic arrival pattern. Choose among 'sync_burst', 'sync_smooth', and 'async'.")
    args.add_argument('--awake-dur', type=float, default=None, help="Flow awake time.")
    args.add_argument('--awake-dist', type=str, default="constant",
                      help="Flow awake time distribution. Choose between 'exponential' and 'constant'.")
    args.add_argument('--sync-jitter', type=float, default=0,
                      help="Jitter for synchronized flow burst. Only active when the arrival pattern is 'sync_burst'.")
    return args.parse_args()


if __name__ == '__main__':
    """An example use case of the traffic shaping network simulator."""
    # Parse the command line arguments.
    args = getargs()
    # Load the input data.
    flow_profile = np.load(args.flow_path)
    flow_route = np.load(args.route_path)
    # Assuming full shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    # Set the random seed for reproducible results.
    np.random.seed(0)
    # Create the output folder if not exist.
    os.makedirs(args.out_dir, exist_ok=True)
    # Establish the simulator. Make your own choice of the input parameters.
    simulator = NetworkSimulator(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                                 scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
                                 buffer_bound=args.buffer_bound, arrival_pattern_type=args.arrival_pattern_type,
                                 awake_dur=args.awake_dur, awake_dist=args.awake_dist, sync_jitter=args.sync_jitter,
                                 arrival_pattern=None, keep_per_hop_departure=False, scaling_factor=1.0,
                                 packet_size=1)
    # Start the simulation.
    start = time.time()
    simulator.simulate()
    time_taken = time.time() - start
    # Check the statistics of interest (e.g., packet arrival time, end-to-end delay, the desired latency target,
    # and the backlog) after the simulation.
    arrival_time = simulator.arrival_time
    end_to_end_delay = simulator.end_to_end_delay
    latency_target = simulator.latency_target
    scheduler_backlog = simulator.scheduler_max_backlog
    ingress_shaper_backlog, shaper_backlog = [], []
    if hasattr(simulator, "ingress_reprofiler_backlog"):
        ingress_shaper_backlog = simulator.ingress_reprofiler_backlog
    if hasattr(simulator, "reprofiler_backlog"):
        shaper_backlog = simulator.reprofiler_backlog
    # You can retrieve other statistics, such as the (normalized) end-to-end delay of each packet.
    delay_aggregate = []
    for flow_end_to_end, flow_delay_target in zip(end_to_end_delay, latency_target):
        for packet_end_to_end in flow_end_to_end:
            # Check whether the packet is lost.
            if packet_end_to_end != -1:
                # Compute normalized delay.
                delay_aggregate.append(packet_end_to_end / flow_delay_target * 100)
            else:
                delay_aggregate.append(-1)
    delay_aggregate = np.array(delay_aggregate)
    # Compute the loss rate.
    loss_mask = (delay_aggregate == -1)
    loss_rate = np.sum(loss_mask) / len(delay_aggregate)
    # Compute the percentage of delay violation (excluding lost packets).
    violation = np.sum(delay_aggregate > 100) / np.sum(np.logical_not(loss_mask))
    # You may save the output data.
    data = {"arrival_time": arrival_time, "end_to_end_delay": end_to_end_delay,
            "latency_target": latency_target, "scheduler_backlog": scheduler_backlog,
            "ingress_shaper_backlog": ingress_shaper_backlog, "shaper_backlog": shaper_backlog, "time": time_taken}
    output_file = os.path.join(args.out_dir, f"{args.file_name}.pickle")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # You can finally use the built-in plotting functionalities to visualize the data and save the results.
    # For example, you may plot the distribution of the (normalized) end-to-end delay.
    plot_delay_distribution(delay_aggregate[np.logical_not(loss_mask)], args.out_dir, "normalized_delay_distribution")
