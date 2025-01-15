import argparse
import numpy as np
import time

from lib.rl_env import RLNetworkEnv

"""Create a reinforcement learning environment based on the network simulator and start a play."""


def getargs():
    """Parse command line arguments."""

    args = argparse.ArgumentParser()
    args.add_argument('flow_path', help="Path to the input npy files describing flow profiles.")
    args.add_argument('route_path', help="Path to the input npy files describing flow routes.")
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
    args.add_argument('--sync-jitter', type=float, default=0,
                      help="Jitter for synchronized flow burst. Only active when the arrival pattern is 'sync_burst'.")
    args.add_argument('--periodic-arrival-ratio', type=float, default=1.0,
                      help="Percent of flows that have periodic arrival patterns. Non-periodic flows send the maximum"
                           "amount of traffic throughout the simulation.")
    args.add_argument('--awake-prob-choice', type=float, default=[1.0], nargs='+',
                      help="Choices of probability that periodic flows awake at the beginning of each period.")
    args.add_argument('--awake-prob-sample-weight', type=float, default=[1.0], nargs='+',
                      help="The sampling weight of each choice of flow awake probability.")
    args.add_argument('--awake-dur', type=float, default=0.0, help="Length of awake time of periodic flows.")
    args.add_argument('--awake-dist', type=str, default="constant",
                      help="Periodic flow awake time distribution. Choose between 'exponential' and 'constant'.")
    args.add_argument('--sleep-dur', type=str, default='max', help="Length of sleep time of periodic flows. Can be set"
                                                                  "to 'min', 'max', or a number.")
    args.add_argument('--sleep-dist', type=str, default="constant",
                      help="Periodic flow sleep time distribution. Choose between 'uniform' and 'constant'.")
    args.add_argument('--pause-interval', type=float, default=1,
                      help="The length of a time step (in second) for the reinforcement learning environment.")
    args.add_argument('--high-reward', type=float, default=1,
                      help="The highest possible reward received by a flow when its end-to-end delay is 0.")
    args.add_argument('--low-reward', type=float, default=0.1,
                      help="The reward received by a flow when its end-to-end delay is equal to the worst case bound.")
    args.add_argument('--penalty', type=float, default=-10,
                      help="The negative penalty received by a flow when its end-to-end delay exceeds the bound.")
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
    # Set the random seed for reproducible results.
    np.random.seed(0)
    # Create the RL environment. Make your own choice of the input parameters.
    environment = RLNetworkEnv(flow_profile, flow_route, shaping_delay, simulation_time=args.simulation_time,
                               scheduling_policy=args.scheduling_policy, shaping_mode=args.shaping_mode,
                               buffer_bound=args.buffer_bound, arrival_pattern_type=args.arrival_pattern_type,
                               sync_jitter=args.sync_jitter, periodic_arrival_ratio=args.periodic_arrival_ratio,
                               awake_prob_choice=tuple(args.awake_prob_choice),
                               awake_prob_sample_weight=tuple(args.awake_prob_sample_weight),
                               awake_dur=args.awake_dur, awake_dist=args.awake_dist, sleep_dur=args.sleep_dur,
                               sleep_dist=args.sleep_dist, arrival_pattern=None, keep_per_hop_departure=False,
                               scaling_factor=1.0, packet_size=1, propagation_delay=0,
                               pause_interval=args.pause_interval, high_reward=args.high_reward,
                               low_reward=args.low_reward, penalty=args.penalty)
    # Initialize the environment and get the initial state.
    initial_state = environment.reset()
    # Keep iterating until the end of an episode.
    while True:
        # Specify the shaping control action. Here we assume we always turn on the shapers.
        # You may train an RL agent to make the shaping control decision here.
        action = np.ones((environment.num_action,), dtype=bool)
        next_state, reward, terminate, truncate = environment.step(action)
        # End the episode if the simulation terminates or gets truncated.
        if terminate or truncate:
            break
