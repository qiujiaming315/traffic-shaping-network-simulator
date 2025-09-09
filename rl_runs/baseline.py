import argparse
import datetime
import os
from pathlib import Path

import numpy as np
from wrapper import make_one_env


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Parameters for the network simulator.
    parser.add_argument('flow_dir', help="Directory to the input npy files describing flow profiles.")
    parser.add_argument('route_dir', help="Directory to the input npy files describing flow routes.")
    parser.add_argument("sim_idx", type=int, default=1, help="Simulation index.")
    parser.add_argument('--simulation-time', type=float, default=100.0, help="Total simulation time (in seconds).")
    parser.add_argument('--traffic-cycle-period', type=float, default=5.0, help="Length of traffic bursting cycle.")
    parser.add_argument('--clock-drift-std', type=float, default=0.01,
                        help="The standard deviation of the normal distribution for clock drift of traffic sources.")
    parser.add_argument('--load-perturbation', type=float, default=[0.05, 0.005], nargs='+',
                        help="The average and standard deviation of the normal distribution for load perturbation.")
    parser.add_argument('--reboot-inter-arrival-avg', type=float, default=200.0,
                        help="The average inter-arrival time of system reboot events.")
    parser.add_argument('--reboot-time-avg', type=float, default=5.0, help="The average time of a system reboot.")
    parser.add_argument('--packet-size', type=float, default=1.0, help="The size of each packet.")
    parser.add_argument('--shaper-backlog-window-size', type=float, default=50.0,
                        help="The sliding window size (in seconds) to collect shaper backlog records.")
    parser.add_argument('--shaper-backlog-top-k', type=int, default=6,
                        help="The number of recent top backlog records to embed in the observations.")
    parser.add_argument('--shaper-num-uniform-samples', type=int, default=10,
                        help="The granularity of uniform sampling of backlog records in local data collection.")
    parser.add_argument('--shaper-min-num-inter-arrival', type=int, default=20,
                        help="The minimum number of burst inter-arrival records to keep at each flow (for TAAD).")
    parser.add_argument('--pause-interval', type=float, default=1,
                        help="The length of a time step (in second) for the reinforcement learning environment.")
    parser.add_argument('--action-mode', type=str, default="time",
                        help="Dynamic shaping control mode. Choose between 'time' and 'prob'.")
    parser.add_argument('--discrete-actions', type=float, default=[0.0001, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0],
                        nargs='+', help="A discrete set of control parameter values for the RL policy to select from.")
    parser.add_argument('--reward-weights', type=float, default=[0.6, 0.4], nargs='+',
                        help="The weights of average reward and delay violation rate in computing reward.")
    parser.add_argument('--average-bounds', type=float, default=[0.0, 1.0], nargs='+',
                        help="The lower and upper bounds of average delay.")
    parser.add_argument('--violation-bounds', type=float, default=[0.0, 1.0], nargs='+',
                        help="The lower and upper bounds of delay violation rate.")
    # Parameters for the RL model.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--same-seed", action='store_true')
    parser.add_argument("--policy", choices=['random', 'fixed_param'],
                        help="Choose a baseline policy: 'random' or 'fixed_param'")
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-name", type=str, default="")
    return parser.parse_args()


def test_baseline(args=get_args()):
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = args.policy
    if args.log_name == "":
        args.log_name = os.path.join(args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, args.log_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if args.policy == "fixed_param":
        log_paths = [os.path.join(log_path, str(t)) for t in args.discrete_actions]
    else:
        log_paths = [log_path]
    for p in log_paths:
        stats_path = os.path.join(p, "stats")
        Path(stats_path).mkdir(parents=True, exist_ok=True)
        trail_paths = [os.path.join(stats_path, str(t)) for t in range(args.test_num)]
        for trail_path in trail_paths:
            Path(trail_path).mkdir(parents=True, exist_ok=True)

    # helper function to carry out evaluation on one environment given policy
    def evaluate_policy(env, policy):
        while True:
            action = policy()
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        return

    if args.policy == "fixed_param":
        for t_idx in range(len(args.discrete_actions)):
            policy = lambda: t_idx
            for e_idx in range(args.test_num):
                # prepare the environments for evaluation
                seed = args.seed if args.same_seed else args.seed + e_idx
                path = os.path.join(log_paths[t_idx], "stats", str(e_idx))
                env = make_one_env(args, log_path=path, save_delay_stats=True, save_action=True, save_reward=True,
                                   seed=seed)
                evaluate_policy(env, policy)
    else:
        for e_idx in range(args.test_num):
            # prepare the environments for evaluation
            seed = args.seed if args.same_seed else args.seed + e_idx
            path = os.path.join(log_path, "stats", str(e_idx))
            env = make_one_env(args, log_path=path, save_delay_stats=True, save_action=True, save_reward=True,
                               seed=seed)
            rng = np.random.default_rng(seed)
            policy = lambda: rng.choice(len(args.discrete_actions))
            evaluate_policy(env, policy)


if __name__ == "__main__":
    test_baseline(get_args())
