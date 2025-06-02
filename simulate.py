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
    args.add_argument('--traffic-cycle-period', type=float, default=5.0, help="Length of traffic bursting cycle.")
    args.add_argument('--clock-drift-std', type=float, default=0.01,
                      help="The standard deviation of the normal distribution for clock drift of traffic sources.")
    args.add_argument('--load-perturbation', type=float, default=[0.05, 0.005], nargs='+',
                      help="The average and standard deviation of the normal distribution for load perturbation.")
    args.add_argument('--reboot-inter-arrival-avg', type=float, default=200.0,
                      help="The average inter-arrival time of system reboot events.")
    args.add_argument('--reboot-time-avg', type=float, default=5.0,
                      help="The average time of a system reboot.")
    args.add_argument('--sleep-dur', type=str, default='max', help="Length of sleep time of periodic flows. Can be set"
                                                                   "to 'min', 'max', or a number.")
    args.add_argument('--sleep-dist', type=str, default="constant",
                      help="Periodic flow sleep time distribution. Choose between 'uniform' and 'constant'.")
    return args.parse_args()


if __name__ == '__main__':
    """An example use case of the traffic shaping network simulator."""
    # Parse the command line arguments.
    args = getargs()
    # Load the input data.
    flow_profile = np.load(args.flow_path)
    flow_route = np.load(args.route_path)
    # Assuming greedy shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    # Set the random seed for reproducible results.
    np.random.seed(0)
    # Create the output folder if not exist.
    os.makedirs(args.out_dir, exist_ok=True)
    # Establish the simulator. Make your own choice of the input parameters.
    simulator = NetworkSimulator(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                                 scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
                                 buffer_bound=args.buffer_bound, traffic_cycle_period=args.traffic_cycle_period,
                                 clock_drift_std=args.clock_drift_std, load_perturbation=tuple(args.load_perturbation),
                                 reboot_inter_arrival_avg=args.reboot_inter_arrival_avg,
                                 reboot_time_avg=args.reboot_time_avg, arrival_pattern=None, passive_tb=True,
                                 keep_per_hop_departure=False, repeat=False, scaling_factor=1.0, packet_size=1,
                                 scheduler_busy_period_window_size=0, scheduler_max_backlog_window_size=0,
                                 propagation_delay=0)
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
