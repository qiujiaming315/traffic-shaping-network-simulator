import bisect
import copy
import heapq
import numpy as np
import gymnasium as gym

from lib.network_simulator import NetworkSimulator, Event, EventType


class RLNetworkEnv(gym.Env):
    """A network environment for RL sampling following the Gym's API. Currently only support FIFO with ingress
    shaping using DeSyncExtraTokenBucket shapers."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, buffer_bound="infinite",
                 arrival_pattern_type="sync", sync_jitter=(0,), sync_jitter_weight=(1.0,), periodic_arrival_ratio=1.0,
                 periodic_pattern_dist=((0.8, 0.1, 0.1),), periodic_pattern_dist_weight=(1.0,), awake_dur=0,
                 awake_dist="constant", sleep_dur="max", sleep_dist="constant", arrival_pattern=None,
                 keep_per_hop_departure=True, repeat=False, scaling_factor=1.0, packet_size=1,
                 scheduler_busy_period_window_size=0, scheduler_max_backlog_window_size=0, propagation_delay=0,
                 tor=0.003, shaper_backlog_window_size=1, shaper_backlog_top_k=10, shaper_num_uniform_samples=10,
                 pause_interval=1, action_mode="time", action_type="discrete",
                 discrete_actions=(0.0001, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0), reward_weights=(0.6, 0.4),
                 average_bounds=(0.0, 1.0), violation_bounds=(0.0, 1.0)):
        scheduling_policy, shaping_mode, passive_tb = "fifo", "is", False
        self.simulator = NetworkSimulator(flow_profile, flow_path, reprofiling_delay, simulation_time=simulation_time,
                                          scheduling_policy=scheduling_policy, shaping_mode=shaping_mode,
                                          buffer_bound=buffer_bound, arrival_pattern_type=arrival_pattern_type,
                                          sync_jitter=sync_jitter, sync_jitter_weight=sync_jitter_weight,
                                          periodic_arrival_ratio=periodic_arrival_ratio,
                                          periodic_pattern_dist=periodic_pattern_dist,
                                          periodic_pattern_dist_weight=periodic_pattern_dist_weight,
                                          awake_dur=awake_dur, awake_dist=awake_dist, sleep_dur=sleep_dur,
                                          sleep_dist=sleep_dist, arrival_pattern=arrival_pattern, passive_tb=passive_tb,
                                          keep_per_hop_departure=keep_per_hop_departure, repeat=repeat,
                                          scaling_factor=scaling_factor, packet_size=packet_size,
                                          scheduler_busy_period_window_size=scheduler_busy_period_window_size,
                                          scheduler_max_backlog_window_size=scheduler_max_backlog_window_size,
                                          propagation_delay=propagation_delay, tor=tor)
        self.repeat = repeat
        self.shaper_backlog_window_size = shaper_backlog_window_size
        self.shaper_backlog_top_k = shaper_backlog_top_k
        self.shaper_num_uniform_samples = shaper_num_uniform_samples
        self.pause_interval = pause_interval
        valid_action_mode = action_mode in ["time", "prob"]
        assert valid_action_mode, "Please choose an action mode between 'time' and 'prob'."
        self.action_mode = action_mode
        valid_action_type = action_type in ["continuous", "discrete"]
        assert valid_action_type, "Please choose an action type between 'continuous' and 'discrete'."
        self.action_type = action_type
        self.discrete_actions = np.array([])
        if self.action_type == "discrete":
            self.discrete_actions = np.array(discrete_actions)
            assert len(self.discrete_actions) > 0 and np.all(self.discrete_actions > 0), "Please set the discrete " \
                                                                                         "actions to positive values."
        self.reward_weights = np.array(reward_weights)
        assert len(reward_weights) == 2 and np.all(self.reward_weights >= 0), "Please set the weight of average " \
                                                                              "delay and delay violation rate in " \
                                                                              "computing reward to a non-negative " \
                                                                              "value."
        self.average_bounds = np.array(average_bounds)
        assert len(average_bounds) == 2 and 0 <= average_bounds[0] \
               <= average_bounds[1] <= 1, "Please set the lower and upper bounds of (normalized) average delay to " \
                                          "values in [0, 1]."
        self.violation_bounds = np.array(violation_bounds)
        assert len(violation_bounds) == 2 and 0 <= violation_bounds[0] \
               <= violation_bounds[1] <= 1, "Please set the lower and upper bounds of delay violation rate to values " \
                                            "in [0, 1]."
        self.num_flow = self.simulator.num_flow
        self.time = 0
        # Declare the observation and action space.
        self.observation_space = gym.spaces.Dict({
            "shaper_backlog_ratio_history": gym.spaces.Box(low=0.0, high=max(1.0, simulation_time),
                                                           shape=(self.shaper_backlog_top_k, 2), dtype=float),
            "token_ratio": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "scheduler_backlog": gym.spaces.Box(low=0, high=2 ** 63 - 2, shape=(self.simulator.num_link,), dtype=int),
            "scheduler_max_backlog": gym.spaces.Box(low=0, high=2 ** 63 - 2, shape=(self.simulator.num_link,),
                                                    dtype=int),
            "scheduler_utilization": gym.spaces.Box(low=0.0, high=1.0, shape=(self.simulator.num_link,), dtype=float)
        })
        if self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=float)
        # Configure the shapers.
        for flow_idx in range(self.simulator.num_flow):
            for tb in self.simulator.ingress_reprofilers[flow_idx].token_buckets:
                tb.backlog_window_size = self.pause_interval
                tb.num_uniform_samples = self.shaper_num_uniform_samples
                # Disable the unused control mode.
                if self.action_mode == "time":
                    tb.extra_token_prob = tb.latency_target / tb.latency_min + 1
                else:
                    tb.average_wait_time_multiplier = 0
        # Keep track of the backlog state of flows.
        self.shaper_backlog_state = np.zeros((self.shaper_num_uniform_samples, self.simulator.num_flow, 2), dtype=int)
        self.shaper_backlog_ratio_peak_times = list()
        self.shaper_backlog_ratio_peak = dict()
        # Add the first summary event.
        event = Event(pause_interval, EventType.SUMMARY)
        heapq.heappush(self.simulator.event_pool, event)
        # Keep the initial event pool for restoration upon resetting if repeatable.
        self.event_pool_copy = None
        if self.repeat:
            self.event_pool_copy = copy.deepcopy(self.simulator.event_pool)
        return

    def reward_function(self, prev_packet_count_shaper, next_packet_count_shaper):
        all_packets = True
        # Check if there is at least one packet to compute reward.
        if all([p == n for p, n in zip(prev_packet_count_shaper, next_packet_count_shaper)]):
            return 1, all_packets
        # Check if all packets of interest have arrived at the destination.
        if not all([s <= t for s, t in zip(next_packet_count_shaper, self.simulator.packet_count_terminal)]):
            all_packets = False
            next_packet_count_shaper_truncated = [min(s, t) for s, t in
                                                  zip(next_packet_count_shaper, self.simulator.packet_count_terminal)]
            next_packet_count_shaper = copy.deepcopy(next_packet_count_shaper_truncated)
        aggregate_normalized_delay = []
        for flow_idx, (old_count, new_count) in enumerate(zip(prev_packet_count_shaper, next_packet_count_shaper)):
            for packet_number in range(old_count, new_count):
                packet_end_to_end = self.simulator.end_to_end_delay[flow_idx][packet_number]
                if packet_end_to_end != -1:
                    packet_end_to_end /= self.simulator.latency_target[flow_idx]
                aggregate_normalized_delay.append(packet_end_to_end)
        if len(aggregate_normalized_delay) == 0:
            return 1, all_packets
        aggregate_normalized_delay = np.array(aggregate_normalized_delay)
        average_delay = np.mean(aggregate_normalized_delay)
        violation_rate = np.sum(aggregate_normalized_delay > 1) / len(aggregate_normalized_delay)
        # Clip the average delay and delay violation rate.
        average_score = (self.average_bounds[1] - average_delay) / (self.average_bounds[1] - self.average_bounds[0])
        average_score = max(min(average_score, 1), 0)
        violation_score = (self.violation_bounds[1] - violation_rate) / (
                self.violation_bounds[1] - self.violation_bounds[0])
        violation_score = max(min(violation_score, 1), 0)
        reward = self.reward_weights[0] * average_score + self.reward_weights[1] * violation_score
        return reward, all_packets

    def get_top_k_shaper_ratio_peak(self):
        # Discard outdated records.
        window_idx = bisect.bisect_left(self.shaper_backlog_ratio_peak_times,
                                        self.time - self.shaper_backlog_window_size)
        for time in self.shaper_backlog_ratio_peak_times[:window_idx]:
            del self.shaper_backlog_ratio_peak[time]
        self.shaper_backlog_ratio_peak_times = self.shaper_backlog_ratio_peak_times[window_idx:]
        # Retrieve the recent k peak backlog ratio values.
        top_records = [(self.time - record_time, self.shaper_backlog_ratio_peak[record_time]) for record_time in
                       self.shaper_backlog_ratio_peak_times[-1:-self.shaper_backlog_top_k - 1:-1]]
        # Zerp-padding if not enough history records are available.
        for _ in range(self.shaper_backlog_top_k - len(top_records)):
            top_records.append((0, 0))
        return np.array(top_records)

    def get_obs(self):
        # Collect the history of ingress shaper backlog states.
        for flow_idx in range(self.simulator.num_flow):
            for tb_idx, tb in enumerate(self.simulator.ingress_reprofilers[flow_idx].token_buckets):
                backlog_records = tb.peek_backlog_samples(self.time)
                backlog_old = self.shaper_backlog_state[self.shaper_num_uniform_samples - 1, flow_idx, tb_idx]
                record_idx = 0
                for sample_idx in range(self.shaper_num_uniform_samples):
                    if record_idx < len(backlog_records) and sample_idx == backlog_records[record_idx][0]:
                        backlog_old = backlog_records[record_idx][1]
                        record_idx += 1
                    self.shaper_backlog_state[sample_idx, flow_idx, tb_idx] = backlog_old
        # Restore the shaper backlog peak values.
        sample_window_size = self.pause_interval / self.shaper_num_uniform_samples
        for sample_idx in range(self.shaper_num_uniform_samples):
            sample_time = sample_idx * sample_window_size + self.time - self.pause_interval
            sample_backlog = np.amax(self.shaper_backlog_state[sample_idx], axis=1)
            if not np.all(sample_backlog == 0):
                self.shaper_backlog_ratio_peak_times.append(sample_time)
                self.shaper_backlog_ratio_peak[sample_time] = np.sum(sample_backlog) / np.sum(
                    self.simulator.token_bucket_profile[:, 1])
        # Retrieve the top-k shaper backlog ratio peak values.
        shaper_backlog_ratio_peak_top = self.get_top_k_shaper_ratio_peak()
        # Compute the ratio of unused tokens.
        unused_token_num, total_token_num = 0, 0
        for flow_idx, tb in enumerate(self.simulator.token_buckets):
            flow_packet_size = self.simulator.packet_size[flow_idx]
            token_num = tb.peek(self.time)
            unused_token_num += token_num * flow_packet_size
            total_token_num += tb.burst * flow_packet_size
        unused_token_ratio = unused_token_num / total_token_num
        # Link scheduler backlog and utilization.
        scheduler_backlog = [0] * self.simulator.num_link
        scheduler_max_backlog = [0] * self.simulator.num_link
        scheduler_utilization = [0] * self.simulator.num_link
        for link_idx in range(self.simulator.num_link):
            sb, su = self.simulator.schedulers[link_idx].peek(self.time)
            max_backlog = self.simulator.schedulers[link_idx].peek_recent_max_backlog(self.time)
            scheduler_backlog[link_idx] = sb
            scheduler_max_backlog[link_idx] = max_backlog
            scheduler_utilization[link_idx] = su
        return {"shaper_backlog_ratio_history": shaper_backlog_ratio_peak_top, "token_ratio": unused_token_ratio,
                "scheduler_backlog": scheduler_backlog, "scheduler_max_backlog": scheduler_max_backlog,
                "scheduler_utilization": scheduler_utilization}

    def get_info(self, all_packets=True):
        return {"reward_on_all_packets": all_packets,
                "packet_count_shaper": copy.deepcopy(self.simulator.packet_count_shaper)}

    def step(self, action):

        # The control activation function.
        def update_wait_time(reprofiler, average_wait_time_multiplier):
            for tb_idx in range(len(reprofiler.token_buckets)):
                tb = reprofiler.token_buckets[tb_idx]
                tb.average_wait_time_multiplier = average_wait_time_multiplier
            return

        def update_extra_token_prob(reprofiler, extra_token_prob):
            for tb_idx in range(len(reprofiler.token_buckets)):
                tb = reprofiler.token_buckets[tb_idx]
                tb.extra_token_prob = extra_token_prob
            return

        # Choose the activation function.
        act_func = update_wait_time if self.action_mode == "time" else update_extra_token_prob
        # Convert the action if needed.
        if self.action_type == "discrete":
            action = self.discrete_actions[action]
        # Expand the global control action to all flows.
        action = np.ones((self.simulator.num_flow,), dtype=float) * action
        # Enforce the reprofiling control actions.
        for flow_idx, a in enumerate(action):
            act_func(self.simulator.ingress_reprofilers[flow_idx], a)
        prev_packet_count_shaper = copy.deepcopy(self.simulator.packet_count_shaper)
        self.time += self.pause_interval
        # Start the simulation.
        self.simulator.simulate()
        next_packet_count_shaper = copy.deepcopy(self.simulator.packet_count_shaper)
        # Add the next summary event.
        next_time = self.time + self.pause_interval
        if next_time <= self.simulator.simulation_time:
            event = Event(next_time, EventType.SUMMARY)
            heapq.heappush(self.simulator.event_pool, event)
        # Check the network status.
        obs = self.get_obs()
        # Compute the reward.
        reward, all_packets = self.reward_function(prev_packet_count_shaper, next_packet_count_shaper)
        # Check if the episode terminates.
        terminated = len(self.simulator.event_pool) == 0
        info = self.get_info(all_packets)
        return obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        # Reset the random number generator.
        super().reset(seed=seed)
        # Reset the simulator.
        arrival_pattern = None if options is None else options["arrival_pattern"]
        self.simulator.reset(arrival_pattern=arrival_pattern)
        self.time = 0
        # Configure the shapers.
        for flow_idx in range(self.simulator.num_flow):
            for tb in self.simulator.ingress_reprofilers[flow_idx].token_buckets:
                tb.backlog_window_size = self.pause_interval
                tb.num_uniform_samples = self.shaper_num_uniform_samples
                # Disable the unused control mode.
                if self.action_mode == "time":
                    tb.extra_token_prob = tb.latency_target / tb.latency_min + 1
                else:
                    tb.average_wait_time_multiplier = 0
        # Keep track of the backlog state of flows.
        self.shaper_backlog_state = np.zeros((self.shaper_num_uniform_samples, self.simulator.num_flow, 2), dtype=int)
        self.shaper_backlog_ratio_peak_times = list()
        self.shaper_backlog_ratio_peak = dict()
        # Restore the initial event pool if repeating the previous episode.
        if self.repeat and arrival_pattern is None:
            self.simulator.event_pool = copy.deepcopy(self.event_pool_copy)
        else:
            # Add the first summary event.
            event = Event(self.pause_interval, EventType.SUMMARY)
            heapq.heappush(self.simulator.event_pool, event)
        # Set the initial state.
        initial_obs = self.get_obs()
        info = self.get_info()
        return initial_obs, info
