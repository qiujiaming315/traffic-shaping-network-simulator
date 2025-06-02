import os

import numpy as np
from lib.rl_env import RLNetworkEnv

from tianshou.env import SubprocVectorEnv


def make_env(args):
    env = make_one_env(args, args.seed)
    train_envs = SubprocVectorEnv([lambda: make_one_env(args, args.seed + s + 1) for s in range(args.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: make_one_env(args, args.seed + args.training_num + s + 1) for s in range(args.test_num)])
    return env, train_envs, test_envs


def make_one_env(args, seed=None):
    # Load the input data.
    flow_profile = np.load(os.path.join(args.flow_dir, f"flow{args.sim_idx}.npy"))
    flow_route = np.load(os.path.join(args.route_dir, f"route{args.sim_idx}.npy"))
    # Assuming full shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    env = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                       buffer_bound="infinite", traffic_cycle_period=args.traffic_cycle_period,
                       clock_drift_std=args.clock_drift_std, load_perturbation=tuple(args.load_perturbation),
                       reboot_inter_arrival_avg=args.reboot_inter_arrival_avg, reboot_time_avg=args.reboot_time_avg,
                       arrival_pattern=None, keep_per_hop_departure=False, scaling_factor=1.0, packet_size=1,
                       propagation_delay=0, shaper_backlog_window_size=args.shaper_backlog_window_size,
                       shaper_backlog_top_k=args.shaper_backlog_top_k,
                       shaper_num_uniform_samples=args.shaper_num_uniform_samples,
                       shaper_max_num_inter_arrival=args.shaper_max_num_inter_arrival, shaper_local_protection_on=False,
                       scheduler_busy_period_window_size=args.pause_interval,
                       scheduler_max_backlog_window_size=args.pause_interval, pause_interval=args.pause_interval,
                       action_mode=args.action_mode, action_type="discrete",
                       discrete_actions=tuple(args.discrete_actions), reward_weights=tuple(args.reward_weights),
                       average_bounds=tuple(args.average_bounds), violation_bounds=tuple(args.violation_bounds),
                       seed=seed)
    return env
