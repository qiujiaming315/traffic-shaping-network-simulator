import copy
import os
import numpy as np
import pickle

from tianshou.env import SubprocVectorEnv

from lib.rl_env import RLNetworkEnv


def make_env(args, log_dir, continuous=False, same_seed=True):
    env = make_one_env(args, continuous=continuous, seed=args.seed)
    log_paths = [os.path.join(log_dir, str(s)) for s in range(args.test_num)]
    if same_seed:
        train_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, seed=args.seed) for _ in range(args.training_num)])
        test_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, log_path=log_paths[s], save_inter_arrival=True,
                                  seed=args.seed) for s in range(args.test_num)])
    else:
        train_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, seed=args.seed + s + 1) for s in
             range(args.training_num)])
        test_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, log_path=log_paths[s], save_inter_arrival=True,
                                  seed=args.seed + args.training_num + s + 1) for s in range(args.test_num)])
    return env, train_envs, test_envs


def make_eval_env(args, log_dir, continuous=False, enable_taad=False, same_seed=True):
    log_paths = [os.path.join(log_dir, str(s)) for s in range(args.test_num)]
    if same_seed:
        eval_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, log_path=log_paths[s], enable_taad=enable_taad,
                                  save_delay_stats=True, save_action=True, seed=args.seed)
             for s in range(args.test_num)])
    else:
        eval_envs = SubprocVectorEnv(
            [lambda: make_one_env(args, continuous=continuous, log_path=log_paths[s], enable_taad=enable_taad,
                                  save_delay_stats=True, save_action=True, seed=args.seed + args.training_num + s + 1)
             for s in range(args.test_num)])
    return eval_envs


def make_one_env(args, continuous=False, log_path="", enable_taad=False, save_inter_arrival=False,
                 save_delay_stats=False, save_action=False, seed=None):
    # Load the input data.
    flow_profile = np.load(os.path.join(args.flow_dir, f"flow{args.sim_idx}.npy"))
    flow_route = np.load(os.path.join(args.route_dir, f"route{args.sim_idx}.npy"))
    # Assuming full shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    if continuous:
        verbose_obs, action_type, discrete_actions = False, "continuous", None
    else:
        verbose_obs, action_type, discrete_actions = True, "discrete", tuple(args.discrete_actions)
    env = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                       buffer_bound="infinite", traffic_cycle_period=args.traffic_cycle_period,
                       clock_drift_std=args.clock_drift_std, load_perturbation=tuple(args.load_perturbation),
                       reboot_inter_arrival_avg=args.reboot_inter_arrival_avg, reboot_time_avg=args.reboot_time_avg,
                       arrival_pattern=None, keep_per_hop_departure=False, scaling_factor=1.0,
                       packet_size=args.packet_size, propagation_delay=0,
                       shaper_backlog_window_size=args.shaper_backlog_window_size,
                       shaper_backlog_top_k=args.shaper_backlog_top_k,
                       shaper_num_uniform_samples=args.shaper_num_uniform_samples,
                       shaper_min_num_inter_arrival=args.shaper_min_num_inter_arrival,
                       shaper_local_protection_on=enable_taad, scheduler_busy_period_window_size=args.pause_interval,
                       scheduler_max_backlog_window_size=args.pause_interval, pause_interval=args.pause_interval,
                       verbose_obs=verbose_obs, action_mode=args.action_mode, action_type=action_type,
                       discrete_actions=discrete_actions, reward_weights=tuple(args.reward_weights),
                       average_bounds=tuple(args.average_bounds), violation_bounds=tuple(args.violation_bounds),
                       log_path=log_path, save_inter_arrival=save_inter_arrival, save_delay_stats=save_delay_stats,
                       save_action=save_action, seed=seed)
    if enable_taad:
        # Load the inter-arrival records.
        inter_arrival_records_path = os.path.join(log_path, "inter_arrival_records.pickle")
        assert os.path.isfile(inter_arrival_records_path), "Please ensure the inter-arrival records exist " \
                                                           "before evaluation if choose to enable taad."
        with open(inter_arrival_records_path, 'rb') as f:
            inter_arrival_records = pickle.load(f)
        # Load the inter-arrival records to the evaluation environments.
        for rp, tb_records in zip(env.simulator.ingress_reprofilers, inter_arrival_records):
            for tb, records in zip(rp.token_buckets, tb_records):
                tb.burst_inter_arrival_records = copy.deepcopy(records)
    return env
