import numpy as np
import os

from lib.network_env import NetworkEnv
from viz.packet_delay_demo import plot_delay_distribution


def get_latency_distribution(flow_dir, route_dir, num_episodes, plot_path, plot_name, simulation_time=100.0,
                             awake_dur=10.0, shaper_mode="on", traffic_pattern="sync"):
    delay_aggregate = []
    flow_route = np.load(os.path.join(route_dir, f"route1.npy"))
    for i in range(num_episodes):
        flow_profile = np.load(os.path.join(flow_dir, f"flow{i + 1}.npy"))
        reprofiling_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
        # Establish the network environment.
        env = NetworkEnv(flow_profile, flow_route, reprofiling_delay, simulation_time, terminate_time=simulation_time,
                         pattern_type=traffic_pattern, awake_dur=awake_dur)
        # Start the simulation.
        _ = env.reset()
        done = False
        while not done:
            actions = np.ones((env.num_flow,)) if shaper_mode == "on" else np.zeros((env.num_flow,))
            _, _, terminated, truncated = env.step(actions)
            done = terminated
        arrival_time = env.arrival_time
        departure_time = env.departure_time
        latency_target = env.latency_target
        end_to_end_delay = [[(d[-1] - a) / target * 100 for a, d in zip(arrival, departure)] for
                            arrival, departure, target in zip(arrival_time, departure_time, latency_target)]
        for flow_delay in end_to_end_delay:
            delay_aggregate.extend(flow_delay)
    delay_aggregate = np.array(delay_aggregate)
    plot_delay_distribution(delay_aggregate, plot_path, plot_name)
    return delay_aggregate


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
    num_flow = 25
    flow_dir = f"./data/flow/heterogeneous/{num_flow}/"
    route_dir = f"./data/route/{num_flow}/"
    output_path = "./figures/heterogeneous_flow/sync_burst_2/"
    os.makedirs(output_path, exist_ok=True)
    np.random.seed(0)
    simulation_time = 10
    awake_dur = 0.5
    end_to_end_on = get_latency_distribution(flow_dir, route_dir, 10, output_path, "shaper_on",
                                             simulation_time=simulation_time, shaper_mode="on",
                                             traffic_pattern="sync_burst", awake_dur=awake_dur)
    end_to_end_off = get_latency_distribution(flow_dir, route_dir, 10, output_path, "shaper_off",
                                              simulation_time=simulation_time, shaper_mode="off",
                                              traffic_pattern="sync_burst", awake_dur=awake_dur)
    critical_percentile = get_critical_percentile(end_to_end_on, end_to_end_off)
    print(f"critical percentile: {critical_percentile * 100: .2f} %")
