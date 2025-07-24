import bisect
import copy
import heapq
import numpy as np
import gymnasium as gym
import os
import pickle

from lib.network_simulator import NetworkSimulator, Event, EventType


class RLNetworkEnv(gym.Env):
    """A network environment for RL sampling following the Gym's API. Currently only support FIFO with ingress
    shaping using DeSyncExtraTokenBucket shapers."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, buffer_bound="infinite",
                 traffic_cycle_period=5.0, clock_drift_std=0.01, load_perturbation=(0.05, 0.005),
                 reboot_inter_arrival_avg=200.0, reboot_time_avg=5.0, arrival_pattern=None,
                 keep_per_hop_departure=True, repeat=False, scaling_factor=1.0, packet_size=1, propagation_delay=0,
                 tor=0.003, shaper_backlog_window_size=50.0, shaper_backlog_top_k=6, shaper_num_uniform_samples=10,
                 shaper_min_num_inter_arrival=20, shaper_local_protection_on=True, shaper_local_protection_time=10.0,
                 scheduler_busy_period_window_size=0, scheduler_max_backlog_window_size=0, pause_interval=1,
                 verbose_obs=True, action_mode="time", action_type="discrete",
                 discrete_actions=(0.0001, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0), reward_weights=(0.6, 0.4),
                 average_bounds=(0.0, 1.0), violation_bounds=(0.0, 1.0), log_path="", save_inter_arrival=False,
                 save_delay_stats=False, save_action=False, seed=None):
        scheduling_policy, shaping_mode, passive_tb = "fifo", "is", False
        self.simulator = NetworkSimulator(flow_profile, flow_path, reprofiling_delay, simulation_time=simulation_time,
                                          scheduling_policy=scheduling_policy, shaping_mode=shaping_mode,
                                          buffer_bound=buffer_bound, traffic_cycle_period=traffic_cycle_period,
                                          clock_drift_std=clock_drift_std, load_perturbation=load_perturbation,
                                          reboot_inter_arrival_avg=reboot_inter_arrival_avg,
                                          reboot_time_avg=reboot_time_avg,
                                          arrival_pattern=arrival_pattern, passive_tb=passive_tb,
                                          keep_per_hop_departure=keep_per_hop_departure, repeat=repeat,
                                          scaling_factor=scaling_factor, packet_size=packet_size,
                                          propagation_delay=propagation_delay, tor=tor, seed=seed)
        self.repeat = repeat
        self.shaper_backlog_window_size = shaper_backlog_window_size
        self.shaper_backlog_top_k = shaper_backlog_top_k
        self.shaper_num_uniform_samples = shaper_num_uniform_samples
        self.shaper_min_num_inter_arrival = shaper_min_num_inter_arrival
        self.shaper_local_protection_on = shaper_local_protection_on
        self.shaper_local_protection_time = shaper_local_protection_time
        self.scheduler_busy_period_window_size = scheduler_busy_period_window_size
        self.scheduler_max_backlog_window_size = scheduler_max_backlog_window_size
        self.pause_interval = pause_interval
        self.verbose_obs = verbose_obs
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
        self.log_path = log_path
        self.save_inter_arrival = save_inter_arrival
        self.save_delay_stats = save_delay_stats
        self.save_action = save_action
        if any([save_inter_arrival, save_delay_stats, save_action]):
            assert log_path != "", "Please specify a path to save the collected statistics."
        self.num_flow = self.simulator.num_flow
        self.time = 0
        # Declare the observation and action space.
        if self.verbose_obs:
            self.observation_space = gym.spaces.Dict({
                "shaper_backlog_ratio_history": gym.spaces.Box(low=0.0, high=max(1.0, simulation_time),
                                                               shape=(self.shaper_backlog_top_k, 2), dtype=float),
                "token_ratio": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
                "scheduler_backlog": gym.spaces.Box(low=0, high=2 ** 63 - 2, shape=(self.simulator.num_link,),
                                                    dtype=int),
                "scheduler_max_backlog": gym.spaces.Box(low=0, high=2 ** 63 - 2, shape=(self.simulator.num_link,),
                                                        dtype=int),
                "scheduler_utilization": gym.spaces.Box(low=0.0, high=1.0, shape=(self.simulator.num_link,),
                                                        dtype=float)
            })
        else:
            self.observation_space = gym.spaces.Box(low=0.0, high=max(1.0, simulation_time),
                                                    shape=(2 * self.shaper_backlog_top_k + 1,), dtype=float)
        if self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = gym.spaces.Box(low=0.0001, high=10.0, shape=(1,), dtype=float)
        # Configure the shapers.
        for flow_idx in range(self.simulator.num_flow):
            for tb in self.simulator.ingress_reprofilers[flow_idx].token_buckets:
                tb.backlog_window_size = self.pause_interval
                tb.num_uniform_samples = self.shaper_num_uniform_samples
                tb.min_num_inter_arrival_collect = self.shaper_min_num_inter_arrival
                tb.local_protection_on = self.shaper_local_protection_on
                tb.local_protection_time = self.shaper_local_protection_time
                # Disable the unused control mode.
                if self.action_mode == "time":
                    tb.extra_token_prob = tb.latency_target / tb.latency_min + 1
                else:
                    tb.average_wait_time_multiplier = 0
        # Configure the schedulers.
        for link_idx in range(self.simulator.num_link):
            link = self.simulator.schedulers[link_idx]
            link.busy_period_window_size = self.scheduler_busy_period_window_size
            link.max_backlog_window_size = self.scheduler_max_backlog_window_size
        # Keep track of the backlog state of flows.
        self.shaper_backlog_state = np.zeros((self.shaper_num_uniform_samples, self.simulator.num_flow, 2), dtype=int)
        self.shaper_backlog_ratio_peak_times = list()
        self.shaper_backlog_ratio_peak = dict()
        # Keep track of the actions taken by the agent.
        self.action_history = []
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
        self.shaper_backlog_state = np.zeros_like(self.shaper_backlog_state)
        for flow_idx in range(self.simulator.num_flow):
            for tb_idx, tb in enumerate(self.simulator.ingress_reprofilers[flow_idx].token_buckets):
                backlog_records = tb.peek_backlog_samples(self.time)
                # backlog_old = self.shaper_backlog_state[self.shaper_num_uniform_samples - 1, flow_idx, tb_idx]
                # record_idx = 0
                # for sample_idx in range(self.shaper_num_uniform_samples):
                #     if record_idx < len(backlog_records) and sample_idx == backlog_records[record_idx][0]:
                #         backlog_old = backlog_records[record_idx][1]
                #         record_idx += 1
                #     self.shaper_backlog_state[sample_idx, flow_idx, tb_idx] = backlog_old
                for sample_idx, sample_backlog in backlog_records:
                    self.shaper_backlog_state[sample_idx, flow_idx, tb_idx] = sample_backlog
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
        if self.verbose_obs:
            return {"shaper_backlog_ratio_history": shaper_backlog_ratio_peak_top, "token_ratio": unused_token_ratio,
                    "scheduler_backlog": scheduler_backlog, "scheduler_max_backlog": scheduler_max_backlog,
                    "scheduler_utilization": scheduler_utilization}
        else:
            return np.append(shaper_backlog_ratio_peak_top.flatten(), unused_token_ratio)

    def get_info(self, all_packets=True):
        return {"reward_on_all_packets": all_packets,
                "packet_count_shaper": copy.deepcopy(self.simulator.packet_count_shaper)}

    def save_inter_arrival_helper(self):
        # Save the inter-arrival records from the simulator.
        file_name = os.path.join(self.log_path, "inter_arrival_records.pickle")
        if not os.path.isfile(file_name):
            inter_arrival_records = []
            for rp in self.simulator.ingress_reprofilers:
                tb_records = []
                for tb in rp.token_buckets:
                    tb_records.append(tb.burst_inter_arrival_records)
                inter_arrival_records.append(tb_records)
            with open(file_name, 'wb') as f:
                pickle.dump(inter_arrival_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def save_delay_stats_helper(self):
        # Save the delay statistics from the simulator.
        file_name = os.path.join(self.log_path, f"delay_stats.pickle")
        if not os.path.isfile(file_name):
            normalized_end_to_end = [[e / flow_target for e in flow_end_to_end if e != -1] for
                                     flow_end_to_end, flow_target in
                                     zip(self.simulator.end_to_end_delay, self.simulator.latency_target)]
            aggregate_normalized_delay = []
            for flow_normalized_delay in normalized_end_to_end:
                aggregate_normalized_delay.extend(flow_normalized_delay)
            aggregate_normalized_delay = np.array(aggregate_normalized_delay)
            average_delay = np.mean(aggregate_normalized_delay)
            worst_delay = np.amax(aggregate_normalized_delay)
            violation_rate = np.sum(aggregate_normalized_delay > 1) / len(aggregate_normalized_delay)
            delay_stats = {"aggregate_normalized_delay": aggregate_normalized_delay, "average_delay": average_delay,
                           "worst_delay": worst_delay, "violation_rate": violation_rate}
            with open(file_name, 'wb') as f:
                pickle.dump(delay_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def save_action_helper(self):
        # Save the action history.
        file_name = os.path.join(self.log_path, f"action.pickle")
        if not os.path.isfile(file_name):
            with open(file_name, 'wb') as f:
                pickle.dump(self.action_history, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

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
        # Record the action history.
        if self.save_action:
            self.action_history.append(action)
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
        truncated = False
        if terminated:
            # Save the collected statistics.
            if self.save_inter_arrival:
                self.save_inter_arrival_helper()
            if self.save_delay_stats:
                self.save_delay_stats_helper()
            if self.save_action:
                self.save_action_helper()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the random number generator.
        super().reset(seed=seed)
        # Reset the simulator.
        arrival_pattern = None if options is None else options["arrival_pattern"]
        self.simulator.reset(seed=seed, arrival_pattern=arrival_pattern)
        self.time = 0
        # Configure the shapers.
        for flow_idx in range(self.simulator.num_flow):
            for tb in self.simulator.ingress_reprofilers[flow_idx].token_buckets:
                tb.backlog_window_size = self.pause_interval
                tb.num_uniform_samples = self.shaper_num_uniform_samples
                tb.min_num_inter_arrival_collect = self.shaper_min_num_inter_arrival
                tb.local_protection_on = self.shaper_local_protection_on
                tb.local_protection_time = self.shaper_local_protection_time
                # Disable the unused control mode.
                if self.action_mode == "time":
                    tb.extra_token_prob = tb.latency_target / tb.latency_min + 1
                else:
                    tb.average_wait_time_multiplier = 0
        # Configure the schedulers.
        for link_idx in range(self.simulator.num_link):
            link = self.simulator.schedulers[link_idx]
            link.busy_period_window_size = self.scheduler_busy_period_window_size
            link.max_backlog_window_size = self.scheduler_max_backlog_window_size
        # Keep track of the backlog state of flows.
        self.shaper_backlog_state = np.zeros((self.shaper_num_uniform_samples, self.simulator.num_flow, 2), dtype=int)
        self.shaper_backlog_ratio_peak_times = list()
        self.shaper_backlog_ratio_peak = dict()
        # Keep track of the actions taken by the agent.
        self.action_history = []
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
