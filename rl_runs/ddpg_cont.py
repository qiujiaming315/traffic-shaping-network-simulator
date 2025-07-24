import argparse
import datetime
import os
from pathlib import Path
import pprint

import numpy as np
import torch
from wrapper import make_env, make_eval_env

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


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
    parser.add_argument('--reward-weights', type=float, default=[0.6, 0.4], nargs='+',
                        help="The weights of average reward and delay violation rate in computing reward.")
    parser.add_argument('--average-bounds', type=float, default=[0.0, 1.0], nargs='+',
                        help="The lower and upper bounds of average delay.")
    parser.add_argument('--violation-bounds', type=float, default=[0.0, 1.0], nargs='+',
                        help="The lower and upper bounds of delay violation rate.")
    # Parameters for the RL model.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--same-seed", action='store_true')
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-name", type=str, default="")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--do-train", action='store_true')
    parser.add_argument("--do-evaluate", action='store_true')
    parser.add_argument("--enable-taad", action='store_true')
    return parser.parse_args()


def test_ddpg(args=get_args()):
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ddpg_cont"
    if args.log_name == "":
        args.log_name = os.path.join(args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, args.log_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    stats_path = os.path.join(log_path, "stats")
    Path(stats_path).mkdir(parents=True, exist_ok=True)
    trail_paths = [os.path.join(stats_path, str(t)) for t in range(args.test_num)]
    for trail_path in trail_paths:
        Path(trail_path).mkdir(parents=True, exist_ok=True)
    # prepare the environments for training and test
    env, train_envs, test_envs = make_env(args, stats_path, continuous=True, same_seed=args.same_seed)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net_a, args.action_shape, max_action=args.max_action, device=args.device).to(
        args.device,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num, random=True)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if args.do_train:
        # logger
        logger_factory = LoggerFactoryDefault()
        logger_factory.logger_type = "tensorboard"

        logger = logger_factory.create_logger(
            log_dir=log_path,
            experiment_name=args.log_name,
            run_id=args.resume_id,
            config_dict=vars(args),
        )
        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    if args.do_evaluate:
        model_path = os.path.join(log_path, "policy.pth")
        # Make sure the saved model weights exist.
        assert os.path.isfile(model_path), "Please ensure the pre-trained model exists before evaluation."
        policy.load_state_dict(torch.load(model_path, map_location=args.device))
        eval_envs = make_eval_env(args, stats_path, continuous=True, enable_taad=args.enable_taad,
                                  same_seed=args.same_seed)
        eval_collector = Collector(policy, eval_envs)
        collect_result = eval_collector.collect(reset_before_collect=True, n_episode=args.test_num)
        collect_result.pprint_asdict()


if __name__ == "__main__":
    test_ddpg(get_args())
