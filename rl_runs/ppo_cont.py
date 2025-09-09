import argparse
import datetime
import os
from pathlib import Path
import pprint

import numpy as np
import torch
from wrapper import make_env, make_eval_env
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic


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
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=10)
    # ppo special
    parser.add_argument("--rew-norm", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
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


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo_cont"
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
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy: BasePolicy) -> None:
        state = {"model": policy.state_dict()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

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
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
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
        for env_activator in eval_envs:
            eval_collector = Collector(policy, env_activator)
            collect_result = eval_collector.collect(reset_before_collect=True, n_episode=1)
            # collect_result.pprint_asdict()


if __name__ == "__main__":
    test_ppo()
