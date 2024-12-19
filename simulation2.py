import numpy as np
import argparse
import os
import pickle
import time

from lib.network_simulator import NetworkSimulator
# from viz.packet_delay_demo import plot_delay_distribution


def load_flow(flow_dir, route_dir, file_idx):
    flow_profile = np.load(os.path.join(flow_dir, f"flow{file_idx}.npy"))
    if os.path.isfile(os.path.join(route_dir, f"route{file_idx}.npy")):
        flow_route = np.load(os.path.join(route_dir, f"route{file_idx}.npy"))
    else:
        assert os.path.isfile(os.path.join(route_dir, f"route{file_idx}.npz"))
        route_data = np.load(os.path.join(route_dir, f"route{file_idx}.npz"))
        flow_route = route_data["routes"]
        # Multiply each flow according to the number of destination node(s) it has.
        flow_profile_expanded = np.zeros((0, flow_profile.shape[1]))
        for flow, dn in zip(flow_profile, route_data["app_dest_num"]):
            flow_profile_expanded = np.concatenate(
                (flow_profile_expanded, np.ones((dn, flow_profile.shape[1])) * flow), axis=0)
        flow_profile = flow_profile_expanded
    return flow_profile, flow_route


def simulation(flow_profile, flow_route, reprofiling_delay, simulation_time=100.0, scheduling_policy="fifo",
               shaping_mode="pfs", buffer_bound="infinite", arrival_pattern_type="sync_burst", awake_dur=10.0,
               awake_dist="exponential", sync_jitter=0, arrival_pattern=None, keep_per_hop_departure=False,
               scaling_factor=1.0, packet_size=1.0):
    simulator = NetworkSimulator(flow_profile, flow_route, reprofiling_delay, simulation_time=simulation_time,
                                 scheduling_policy=scheduling_policy, shaping_mode=shaping_mode,
                                 buffer_bound=buffer_bound, arrival_pattern_type=arrival_pattern_type,
                                 awake_dur=awake_dur, awake_dist=awake_dist, sync_jitter=sync_jitter,
                                 arrival_pattern=arrival_pattern, keep_per_hop_departure=keep_per_hop_departure,
                                 scaling_factor=scaling_factor, packet_size=packet_size)
    # Start the simulation.
    start_time = time.time()
    simulator.simulate()
    time_taken = time.time() - start_time
    end_to_end_delay = simulator.end_to_end_delay
    latency_target = simulator.latency_target
    normalized_end_to_end_delay = [[e / flow_target * 100 for e in flow_end_to_end if e != -1] for
                                   flow_end_to_end, flow_target in zip(end_to_end_delay, latency_target)]
    aggregate_normalized_delay = []
    for flow_normalized_delay in normalized_end_to_end_delay:
        aggregate_normalized_delay.extend(flow_normalized_delay)
    return simulator, time_taken, aggregate_normalized_delay


def batch_simulation(flow_dir, route_dir, save_dir, start, end, simulation_time=100.0,
                     scheduling_policy="fifo", shaping_mode="pfs", buffer_bound="infinite",
                     arrival_pattern_type="sync", awake_dur=10.0, awake_dist="exponential", sync_jitter=0,
                     scaling_factor=1.0, arrival_pattern=None, packet_size=1.0):
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    for i in range(start, end):
        file_name = os.path.join(save_dir, f"result{i + 1}.pickle")
        if not os.path.isfile(file_name) or save_dir == "":
            flow_profile, flow_route = load_flow(flow_dir, route_dir, i + 1)
            reprofiling_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
            # Establish the network simulator.
            arrival = None if arrival_pattern is None else arrival_pattern[i - start]
            simulator, time_taken, aggregate_normalized_delay = simulation(flow_profile, flow_route, reprofiling_delay,
                                                                           simulation_time=simulation_time,
                                                                           scheduling_policy=scheduling_policy,
                                                                           shaping_mode=shaping_mode,
                                                                           buffer_bound=buffer_bound,
                                                                           arrival_pattern_type=arrival_pattern_type,
                                                                           awake_dur=awake_dur,
                                                                           awake_dist=awake_dist,
                                                                           sync_jitter=sync_jitter,
                                                                           arrival_pattern=arrival,
                                                                           keep_per_hop_departure=False,
                                                                           scaling_factor=scaling_factor,
                                                                           packet_size=packet_size)
            arrival_time = simulator.arrival_time
            end_to_end_delay = simulator.end_to_end_delay
            latency_target = simulator.latency_target
            scheduler_backlog = simulator.scheduler_max_backlog
            data = {"arrival_time": arrival_time, "end_to_end_delay": end_to_end_delay,
                    "latency_target": latency_target, "scheduler_backlog": scheduler_backlog, "time": time_taken}
            if scheduling_policy == "fifo":
                if shaping_mode in ["pfs", "ils", "is", "ntb"]:
                    data["ingress_reprofiler_backlog"] = simulator.ingress_reprofiler_max_backlog
                if shaping_mode in ["pfs", "ils", "ntb"]:
                    data["reprofiler_backlog"] = simulator.reprofiler_max_backlog
            if save_dir != "":
                with open(file_name, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(file_name, 'rb') as f:
            #     data = pickle.load(f)
    return


def get_arrival_pattern(flow_dir, route_dir, start, end, simulation_time=100.0, awake_dur=10.0,
                        awake_dist="exponential", sync_jitter=0, arrival_pattern_type="sync", scaling_factor=1.0,
                        packet_size=1.0):
    arrival_pattern = []
    for i in range(start, end):
        flow_profile, flow_route = load_flow(flow_dir, route_dir, i + 1)
        reprofiling_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
        # Establish the network simulator.
        simulator = NetworkSimulator(flow_profile, flow_route, reprofiling_delay, simulation_time=simulation_time,
                                     arrival_pattern_type=arrival_pattern_type, awake_dur=awake_dur,
                                     awake_dist=awake_dist, sync_jitter=sync_jitter, scaling_factor=scaling_factor,
                                     packet_size=packet_size)
        arrival_pattern.append(simulator.arrival_pattern)
    return arrival_pattern


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('flow_dir', help="Path to the input npy files describing flow profiles.")
    args.add_argument('route_dir', help="Path to the input npy files describing flow routes.")
    args.add_argument('out_dir', help="Directory to save results.")
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
    args.add_argument('--scaling-factor', type=float, default=1.0,
                      help="Bandwidth scaling factor.")
    args.add_argument('--start', type=int, default=0, help="Start input file index.")
    args.add_argument('--end', type=int, default=0, help="End input file index.")
    return args.parse_args()


if __name__ == '__main__':
    args = getargs()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)
    # packet_size = 1
    # packet_size = 3e-4
    packet_size = 1.28e-4

    arrival_pattern = get_arrival_pattern(args.flow_dir, args.route_dir, args.start, args.end,
                                          simulation_time=args.simulation_time,
                                          awake_dur=args.awake_dur,
                                          awake_dist=args.awake_dist,
                                          sync_jitter=args.sync_jitter,
                                          arrival_pattern_type=args.arrival_pattern_type,
                                          packet_size=packet_size)

    batch_simulation(args.flow_dir, args.route_dir, args.out_dir, args.start, args.end,
                     simulation_time=args.simulation_time,
                     scheduling_policy=args.scheduling_policy,
                     shaping_mode=args.shaping_mode,
                     buffer_bound=args.buffer_bound,
                     scaling_factor=args.scaling_factor,
                     arrival_pattern=arrival_pattern,
                     packet_size=packet_size)

    # plot_delay_distribution(delay, args.out_dir, f"simulate_{args.simulation_time}")
