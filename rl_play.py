import argparse
import gymnasium as gym
import numpy as np
import tianshou as ts
from tianshou.utils import TensorboardLogger
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from lib.rl_env import RLNetworkEnv

"""Create a reinforcement learning environment based on the network simulator and start a play."""


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    # args.add_argument('flow_path', help="Path to the input npy files describing flow profiles.")
    # args.add_argument('route_path', help="Path to the input npy files describing flow routes.")
    args.add_argument('--simulation-time', type=float, default=100.0,
                      help="Total simulation time (in seconds).")
    args.add_argument('--scheduling-policy', type=str, default="fifo",
                      help="Type of scheduler applied at each hop of the network. Choose between 'fifo' and 'sced'.")
    args.add_argument('--shaping-mode', type=str, default="pfs",
                      help="Type of traffic shapers applied. Choose among 'pfs', 'ils', 'is', 'itb',"
                           "and 'ntb'. Only active when scheduling policy is 'fifo'")
    args.add_argument('--buffer-bound', type=str, default='infinite',
                      help="Link buffer bound. Choose between 'infinite', and 'with_shaping'.")
    args.add_argument('--arrival-pattern-type', type=str, default="sync",
                      help="Type of traffic arrival pattern. Choose between 'sync' and 'async'.")
    args.add_argument('--sync-jitter', type=float, default=[0.0], nargs='+',
                      help="A sequence of jitter for synchronized flow burst. Only active when the arrival pattern is "
                           "'sync'.")
    args.add_argument('--sync-jitter-weight', type=float, default=[1.0], nargs='+',
                      help="The sampling weight of the synchronization jitters.")
    args.add_argument('--periodic-arrival-ratio', type=float, default=1.0,
                      help="Percent of flows that have periodic arrival patterns. Non-periodic flows send the maximum "
                           "amount of traffic throughout the simulation.")
    args.add_argument('--periodic-pattern-dist', type=float, default=[0.8, 0.1, 0.1], nargs='+',
                      help="A sequence of 3-element distributions, each representing the probability of selecting "
                           "'awake', 'sleep', or 'keep awake' as the periodic pattern, respectively.")
    args.add_argument('--periodic-pattern-dist-weight', type=float, default=[1.0], nargs='+',
                      help="The sampling weight of the arrival pattern distributions.")
    args.add_argument('--awake-dur', type=float, default=0.0, help="Length of awake time of periodic flows.")
    args.add_argument('--awake-dist', type=str, default="constant",
                      help="Periodic flow awake time distribution. Choose between 'exponential' and 'constant'.")
    args.add_argument('--sleep-dur', type=str, default='max', help="Length of sleep time of periodic flows. Can be set "
                                                                   "to 'min', 'max', or a number.")
    args.add_argument('--sleep-dist', type=str, default="constant",
                      help="Periodic flow sleep time distribution. Choose between 'uniform' and 'constant'.")
    args.add_argument('--passive-tb', action="store_true", help="Whether extra tokens are granted passively or "
                                                                "proactively.")
    args.add_argument('--pause-interval', type=float, default=1,
                      help="The length of a time step (in second) for the reinforcement learning environment.")
    args.add_argument('--action-mode', type=str, default="add_token",
                      help="The control action mode. Choose between 'add_token' and 'on_off'.")
    args.add_argument('--max-token-add', type=int, default=10,
                      help="Maximum number of token to add to the traffic shapers. Only used when the action mode is "
                           "'add_token'.")
    args.add_argument('--reward-weights', type=float, default=[0.6, 0.4], nargs='+',
                      help="The weights of average reward and delay violation rate in computing reward.")
    return args.parse_args()


if __name__ == '__main__':
    """An example use case of the reinforcement learning environment for the network simulator."""
    # Parse the command line arguments.
    args = getargs()
    # Load the input data.
    flow_profile = np.load(args.flow_path)
    flow_route = np.load(args.route_path)
    # Assuming full shaping, compute the shaping delay.
    shaping_delay = np.minimum(flow_profile[:, 2], flow_profile[:, 1] / flow_profile[:, 0])
    # # Set the random seed for reproducible results.
    # np.random.seed(0)
    # # Create the RL environment. Make your own choice of the input parameters.
    # environment = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
    #                            scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
    #                            buffer_bound=args.buffer_bound, arrival_pattern_type=args.arrival_pattern_type,
    #                            sync_jitter=tuple(args.sync_jitter), sync_jitter_weight=tuple(args.sync_jitter_weight),
    #                            periodic_arrival_ratio=args.periodic_arrival_ratio,
    #                            periodic_pattern_dist=np.array(args.periodic_pattern_dist).reshape(-1, 3),
    #                            periodic_pattern_dist_weight=tuple(args.periodic_pattern_dist_weight),
    #                            awake_dur=args.awake_dur, awake_dist=args.awake_dist, sleep_dur=args.sleep_dur,
    #                            sleep_dist=args.sleep_dist, arrival_pattern=None, passive_tb=args.passive_tb,
    #                            keep_per_hop_departure=False, scaling_factor=1.0, packet_size=1,
    #                            scheduler_busy_period_window_size=args.pause_interval, scheduler_max_backlog_window_size=args.pause_interval,
    #                            propagation_delay=0, pause_interval=args.pause_interval, action_mode=args.action_mode,
    #                            max_token_add=args.max_token_add, reward_weights=tuple(args.reward_weights))
    # # Initialize the environment and get the initial state.
    # initial_state = environment.reset()
    # # Keep iterating until the end of an episode.
    # while True:
    #     # Specify the shaping control action. Here we assume we always turn on the shapers.
    #     # You may train an RL agent to make the shaping control decision here.
    #     action = np.ones((environment.simulator.num_flow,), dtype=bool)
    #     next_state, reward, terminate, truncate, _ = environment.step(action)
    #     # End the episode if the simulation terminates or gets truncated.
    #     if terminate or truncate:
    #         break

    env = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                       scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
                       buffer_bound=args.buffer_bound, arrival_pattern_type=args.arrival_pattern_type,
                       sync_jitter=tuple(args.sync_jitter), sync_jitter_weight=tuple(args.sync_jitter_weight),
                       periodic_arrival_ratio=args.periodic_arrival_ratio,
                       periodic_pattern_dist=np.array(args.periodic_pattern_dist).reshape(-1, 3),
                       periodic_pattern_dist_weight=tuple(args.periodic_pattern_dist_weight),
                       awake_dur=args.awake_dur, awake_dist=args.awake_dist, sleep_dur=args.sleep_dur,
                       sleep_dist=args.sleep_dist, arrival_pattern=None, passive_tb=args.passive_tb,
                       keep_per_hop_departure=False, scaling_factor=1.0, packet_size=1,
                       scheduler_busy_period_window_size=args.pause_interval,
                       scheduler_max_backlog_window_size=args.pause_interval,
                       propagation_delay=0, pause_interval=args.pause_interval, action_mode=args.action_mode,
                       max_token_add=args.max_token_add, reward_weights=tuple(args.reward_weights))

    train_envs = gym.make('CartPole-v1')
    test_envs = gym.make('CartPole-v1')


    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape)),
            )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            return logits, state


    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320
    )

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
    ).run()
    # print(f'Finished training! Use {result["duration"]}')

    writer = SummaryWriter('log/dqn')
    logger = TensorboardLogger(writer)

    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 35)
