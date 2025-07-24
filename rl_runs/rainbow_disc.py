import argparse
import datetime
import os
from pathlib import Path
import pprint

import numpy as np
import torch
from network import Rainbow
from wrapper import make_env, make_eval_env

from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import C51Policy, RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer


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
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--n-exploration-decay", type=int, default=1e6)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
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


def test_rainbow(args=get_args()):
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "rainbow_disc"
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
    env, train_envs, test_envs = make_env(args, stats_path, same_seed=args.same_seed)
    args.state_shape = int(np.prod(env.observation_space["shaper_backlog_ratio_history"].shape)) + int(
        np.prod(env.observation_space["token_ratio"].shape))
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = Rainbow(
        args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy: C51Policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        action_space=env.action_space,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.training_num,
            ignore_obs_next=True,
            # save_only_last_obs=True,
            stack_num=1,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.training_num,
            ignore_obs_next=True,
            # save_only_last_obs=True,
            stack_num=1,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm,
        )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec and env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= args.n_exploration_decay:
            eps = args.eps_train - env_step / args.n_exploration_decay * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        # if env_step % 10 == 0:
        #     logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            # if env_step % 10 == 0:
            #     logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

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
        # test train_collector and start filling replay buffer
        train_collector.reset()
        train_collector.collect(n_step=args.batch_size * args.training_num)
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
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
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
        eval_envs = make_eval_env(args, stats_path, enable_taad=args.enable_taad, same_seed=args.same_seed)
        eval_collector = Collector(policy, eval_envs)
        collect_result = eval_collector.collect(reset_before_collect=True, n_episode=args.test_num)
        collect_result.pprint_asdict()


if __name__ == "__main__":
    test_rainbow(get_args())
