import numpy as np
import os

from lib.network_simulator import NetworkSimulator
from lib.rl_env import RLNetworkEnv
from viz.packet_delay_demo import plot_delay_statistics, plot_delay_distribution


def get_latency_distribution(flow_dir, route_dir, num_episodes, simulation_time=100.0, awake_dur=10.0,
                             shaping_mode="per_flow", traffic_pattern="sync", arrival_pattern=None, scaling_factor=1.0,
                             packet_size=1):
    delay_aggregate = []
    flow_route = np.load(os.path.join(route_dir, f"route1.npy"))
    for i in range(num_episodes):
        flow_profile = np.load(os.path.join(flow_dir, f"flow{i + 1}.npy"))
        reprofiling_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
        # Establish the network simulator.
        arrival = None if arrival_pattern is None else arrival_pattern[i]
        simulator = NetworkSimulator(flow_profile, flow_route, reprofiling_delay, simulation_time=simulation_time,
                                     shaping_mode=shaping_mode, arrival_pattern_type=traffic_pattern,
                                     awake_dur=awake_dur, arrival_pattern=arrival, scaling_factor=scaling_factor,
                                     packet_size=packet_size)
        # Start the simulation.
        simulator.simulate()
        arrival_time = simulator.arrival_time
        departure_time = simulator.departure_time
        latency_target = simulator.latency_target
        end_to_end_delay = [[(d[-1] - a) / target * 100 for a, d in zip(arrival, departure)] for
                            arrival, departure, target in zip(arrival_time, departure_time, latency_target)]
        for flow_delay in end_to_end_delay:
            delay_aggregate.extend(flow_delay)
    delay_aggregate = np.array(delay_aggregate)
    return delay_aggregate


def get_arrival_pattern(flow_dir, route_dir, num_episodes, simulation_time=100.0, awake_dur=10.0,
                        traffic_pattern="sync", scaling_factor=1.0, packet_size=1):
    flow_route = np.load(os.path.join(route_dir, f"route1.npy"))
    arrival_pattern = []
    for i in range(num_episodes):
        flow_profile = np.load(os.path.join(flow_dir, f"flow{i + 1}.npy"))
        reprofiling_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
        # Establish the network simulator.
        simulator = NetworkSimulator(flow_profile, flow_route, reprofiling_delay, simulation_time=simulation_time,
                                     arrival_pattern_type=traffic_pattern, awake_dur=awake_dur,
                                     scaling_factor=scaling_factor, packet_size=packet_size)
        arrival_pattern.append(simulator.arrival_pattern)
    return arrival_pattern


def get_bandwidth_scale(flow_dir, route_dir, num_episodes, simulation_time=100.0, awake_dur=10.0,
                        shaping_mode="per_flow", traffic_pattern="sync", arrival_pattern=None):
    low, high = 1.0, 2.0
    delay = get_latency_distribution(flow_dir, route_dir, num_episodes,
                                     simulation_time=simulation_time,
                                     shaping_mode=shaping_mode, traffic_pattern=traffic_pattern,
                                     awake_dur=awake_dur,
                                     arrival_pattern=arrival_pattern)
    if np.sum(delay > 100) == 0:
        return 1
    while True:
        delay = get_latency_distribution(flow_dir, route_dir, num_episodes,
                                         simulation_time=simulation_time,
                                         shaping_mode=shaping_mode, traffic_pattern=traffic_pattern,
                                         awake_dur=awake_dur,
                                         arrival_pattern=arrival_pattern, scaling_factor=high)
        if np.sum(delay > 100) == 0:
            break
        high *= 2
    while True:
        mid = (low + high) / 2
        delay = get_latency_distribution(flow_dir, route_dir, num_episodes,
                                         simulation_time=simulation_time,
                                         shaping_mode=shaping_mode, traffic_pattern=traffic_pattern,
                                         awake_dur=awake_dur,
                                         arrival_pattern=arrival_pattern, scaling_factor=mid)
        if np.sum(delay > 100) == 0:
            high = mid
        else:
            low = mid
        if abs(high - low) < 1e-1:
            break
    return high


def get_critical_percentile(distribution_on, distribution_off):
    distribution_on, distribution_off = np.sort(distribution_on), np.sort(distribution_off)
    low, high = 0.1, 0.9
    while True:
        if distribution_on[int(low * len(distribution_on))] > distribution_off[int(low * len(distribution_off))]:
            break
        low /= 2
    while True:
        if distribution_on[int(high * len(distribution_on))] < distribution_off[int(high * len(distribution_off))]:
            break
        high += (1 - high) / 2
    while True:
        mid = (low + high) / 2
        on_value = distribution_on[int(mid * len(distribution_on))]
        off_value = distribution_off[int(mid * len(distribution_off))]
        if abs(on_value - off_value) < 1e-5 or abs(high - low) < 1e-5:
            break
        if on_value > off_value:
            low = mid
        else:
            high = mid
    return mid


if __name__ == '__main__':
    traffic_pattern = "sync_burst"
    # output_path = f"./figures/heterogeneous_flow/statistics/{traffic_pattern}/"
    output_path = "./figures/heterogeneous_flow/sync_burst_2_test/"
    os.makedirs(output_path, exist_ok=True)
    np.random.seed(0)
    simulation_time = 10
    awake_dur = 0.5
    num_hops = np.arange(1, 11)
    packet_size = 1

    num_hop = 2
    num_flow = (num_hop + 3) * 5
    flow_dir = f"./data/flow/heterogeneous/{num_flow}/"
    route_dir = f"./data/route/{num_flow}/"

    arrival_pattern = get_arrival_pattern(flow_dir, route_dir, 10,
                                          simulation_time=simulation_time,
                                          traffic_pattern=traffic_pattern,
                                          awake_dur=awake_dur, packet_size=packet_size)
    end_to_end_per_flow = get_latency_distribution(flow_dir, route_dir, 10,
                                                   simulation_time=simulation_time,
                                                   shaping_mode="per_flow",
                                                   traffic_pattern=traffic_pattern,
                                                   awake_dur=awake_dur, arrival_pattern=arrival_pattern,
                                                   packet_size=packet_size)
    plot_delay_distribution(end_to_end_per_flow, output_path, "per_flow_shaping")
    end_to_end_interleaved = get_latency_distribution(flow_dir, route_dir, 10,
                                                      simulation_time=simulation_time,
                                                      shaping_mode="interleaved",
                                                      traffic_pattern=traffic_pattern, awake_dur=awake_dur,
                                                      arrival_pattern=arrival_pattern, packet_size=packet_size)
    plot_delay_distribution(end_to_end_interleaved, output_path, "interleaved_shaping")
    end_to_end_ingress = get_latency_distribution(flow_dir, route_dir, 10,
                                                  simulation_time=simulation_time, shaping_mode="ingress",
                                                  traffic_pattern=traffic_pattern, awake_dur=awake_dur,
                                                  arrival_pattern=arrival_pattern, packet_size=packet_size)
    plot_delay_distribution(end_to_end_ingress, output_path, "ingress_shaping")
    end_to_end_no = get_latency_distribution(flow_dir, route_dir, 10,
                                             simulation_time=simulation_time, shaping_mode="none",
                                             traffic_pattern=traffic_pattern, awake_dur=awake_dur,
                                             arrival_pattern=arrival_pattern, packet_size=packet_size)
    plot_delay_distribution(end_to_end_no, output_path, "no_shaping")

    # average, median, ninety_five = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    # ninety_nine, worst, violation = [[], [], [], []], [[], [], [], []], [[], [], [], []]
    # scaling_factor = [[], [], [], []]
    #
    #
    # def add_data(distribution, index):
    #     average[index].append(np.mean(distribution))
    #     sorted_distribution = np.sort(distribution)
    #     median[index].append(sorted_distribution[int(0.5 * len(sorted_distribution))])
    #     ninety_five[index].append(sorted_distribution[int(0.95 * len(sorted_distribution))])
    #     ninety_nine[index].append(sorted_distribution[int(0.99 * len(sorted_distribution))])
    #     worst[index].append(sorted_distribution[-1])
    #     violation[index].append(np.sum(distribution > 100) / len(distribution) * 100)
    #     return
    #
    #
    # for num_hop in num_hops:
    #     num_flow = (num_hop + 3) * 5
    #     flow_dir = f"./data/flow/heterogeneous/{num_flow}/"
    #     route_dir = f"./data/route/{num_flow}/"
    #     arrival_pattern = get_arrival_pattern(flow_dir, route_dir, 10,
    #                                           simulation_time=simulation_time,
    #                                           traffic_pattern=traffic_pattern,
    #                                           awake_dur=awake_dur)
    #     end_to_end_per_flow = get_latency_distribution(flow_dir, route_dir, 10,
    #                                                    simulation_time=simulation_time,
    #                                                    shaping_mode="per_flow",
    #                                                    traffic_pattern=traffic_pattern,
    #                                                    awake_dur=awake_dur, arrival_pattern=arrival_pattern)
    #     add_data(end_to_end_per_flow, 0)
    #     end_to_end_interleaved = get_latency_distribution(flow_dir, route_dir, 10,
    #                                                       simulation_time=simulation_time,
    #                                                       shaping_mode="interleaved",
    #                                                       traffic_pattern=traffic_pattern, awake_dur=awake_dur,
    #                                                       arrival_pattern=arrival_pattern)
    #     add_data(end_to_end_interleaved, 1)
    #     end_to_end_ingress = get_latency_distribution(flow_dir, route_dir, 10,
    #                                                   simulation_time=simulation_time, shaping_mode="ingress",
    #                                                   traffic_pattern=traffic_pattern, awake_dur=awake_dur,
    #                                                   arrival_pattern=arrival_pattern)
    #     add_data(end_to_end_ingress, 2)
    #     end_to_end_no = get_latency_distribution(flow_dir, route_dir, 10,
    #                                              simulation_time=simulation_time, shaping_mode="none",
    #                                              traffic_pattern=traffic_pattern, awake_dur=awake_dur,
    #                                              arrival_pattern=arrival_pattern)
    #     add_data(end_to_end_no, 3)
    # plot_delay_statistics((num_hops, "Number of Hops"), (average, "Delay Average / Delay Bound"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_average")
    # plot_delay_statistics((num_hops, "Number of Hops"), (median, "Delay Median / Delay Bound"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_median")
    # plot_delay_statistics((num_hops, "Number of Hops"), (ninety_five, "Delay 95th / Delay Bound"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_95th")
    # plot_delay_statistics((num_hops, "Number of Hops"), (ninety_nine, "Delay 99th / Delay Bound"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_99th")
    # plot_delay_statistics((num_hops, "Number of Hops"), (worst, "Worst Delay / Delay Bound"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_worst")
    # plot_delay_statistics((num_hops, "Number of Hops"), (violation, "Delay Violation"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "delay_violation")
    #
    # for num_hop in num_hops:
    #     num_flow = (num_hop + 3) * 5
    #     flow_dir = f"./data/flow/heterogeneous/{num_flow}/"
    #     route_dir = f"./data/route/{num_flow}/"
    #
    #     arrival_pattern = get_arrival_pattern(flow_dir, route_dir, 10,
    #                                           simulation_time=simulation_time,
    #                                           traffic_pattern=traffic_pattern,
    #                                           awake_dur=awake_dur)
    #     scaling_factor[0].append(get_bandwidth_scale(flow_dir, route_dir, 10, simulation_time=simulation_time,
    #                                                  shaping_mode="per_flow",
    #                                                  traffic_pattern=traffic_pattern,
    #                                                  awake_dur=awake_dur, arrival_pattern=arrival_pattern))
    #     scaling_factor[1].append(get_bandwidth_scale(flow_dir, route_dir, 10, simulation_time=simulation_time,
    #                                                  shaping_mode="interleaved",
    #                                                  traffic_pattern=traffic_pattern,
    #                                                  awake_dur=awake_dur, arrival_pattern=arrival_pattern))
    #     scaling_factor[2].append(get_bandwidth_scale(flow_dir, route_dir, 10, simulation_time=simulation_time,
    #                                                  shaping_mode="ingress",
    #                                                  traffic_pattern=traffic_pattern,
    #                                                  awake_dur=awake_dur, arrival_pattern=arrival_pattern))
    #     scaling_factor[3].append(get_bandwidth_scale(flow_dir, route_dir, 10, simulation_time=simulation_time,
    #                                                  shaping_mode="none",
    #                                                  traffic_pattern=traffic_pattern,
    #                                                  awake_dur=awake_dur, arrival_pattern=arrival_pattern))
    # scaling_factor = np.array(scaling_factor) * 100
    # plot_delay_statistics((num_hops, "Number of Hops"), (scaling_factor, "Bandwidth Scale"),
    #                       ["Per-flow Shaping", "Interleaved Shaping", "Ingress Shaping", "No Shaping"], output_path,
    #                       "bandwidth_scale")
