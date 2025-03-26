import copy
import heapq
import numpy as np

from lib.network_simulator import NetworkSimulator, Event, EventType


class RLNetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, scheduling_policy="fifo",
                 shaping_mode="is", buffer_bound="infinite", arrival_pattern_type="sync", sync_jitter=0,
                 periodic_arrival_ratio=1.0, periodic_pattern_weight=(0.8, 0.1, 0.1), awake_dur=0,
                 awake_dist="constant", sleep_dur="max", sleep_dist="constant", arrival_pattern=None, passive_tb=False,
                 tb_average_wait_time=0.5, keep_per_hop_departure=True, repeat=False, scaling_factor=1.0, packet_size=1,
                 busy_period_window_size=0, propagation_delay=0, tor=0.003, pause_interval=1, action_mode="add_token",
                 max_token_add=10, high_reward=1, low_reward=0.1, penalty=-10, reward_function_type="linear"):
        self.simulator = NetworkSimulator(flow_profile, flow_path, reprofiling_delay, simulation_time=simulation_time,
                                          scheduling_policy=scheduling_policy, shaping_mode=shaping_mode,
                                          buffer_bound=buffer_bound, arrival_pattern_type=arrival_pattern_type,
                                          sync_jitter=sync_jitter, periodic_arrival_ratio=periodic_arrival_ratio,
                                          periodic_pattern_weight=periodic_pattern_weight, awake_dur=awake_dur,
                                          awake_dist=awake_dist, sleep_dur=sleep_dur, sleep_dist=sleep_dist,
                                          arrival_pattern=arrival_pattern, passive_tb=passive_tb,
                                          tb_average_wait_time=tb_average_wait_time,
                                          keep_per_hop_departure=keep_per_hop_departure, repeat=repeat,
                                          scaling_factor=scaling_factor, packet_size=packet_size,
                                          busy_period_window_size=busy_period_window_size,
                                          propagation_delay=propagation_delay, tor=tor)
        if not passive_tb:
            assert action_mode == "add_token", "The control mode must be 'add_token' to support shapers with " \
                                               "proactively granted extra tokens."
            assert shaping_mode == "is", "The shaping mode must be ingress shaping (is) to support shapers with " \
                                         "proactively granted extra tokens."
        self.passive_tb = passive_tb
        self.repeat = repeat
        self.pause_interval = pause_interval
        valid_mode = action_mode in ["add_token", "on_off"]
        assert valid_mode, "Please choose a control action mode between 'add_token' and 'on_off'."
        self.action_mode = action_mode
        self.max_token_add = max_token_add
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        valid_type = reward_function_type in ["linear", "quadratic"]
        assert valid_type, "Please choose a reward function type between 'linear' and 'quadratic'."
        self.reward_function_type = reward_function_type
        self.time = 0
        self.num_agent = self.simulator.num_flow
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(pause_interval, simulation_time + pause_interval, pause_interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.simulator.event_pool, event)
        # Keep the initial event pool for restoration upon resetting if repeatable.
        self.event_pool_copy = None
        if self.repeat:
            self.event_pool_copy = copy.deepcopy(self.simulator.event_pool)
        return

    def reward_function(self, normalized_delay):
        # Compute the reward given an end-to-end delay
        if normalized_delay != -1 and normalized_delay <= 1:
            if self.reward_function_type == "linear":
                reward = self.low_reward + (1 - normalized_delay) * (self.high_reward - self.low_reward)
            else:
                reward = self.low_reward + (1 - normalized_delay ** 2) * (self.high_reward - self.low_reward)
        else:
            reward = self.penalty
        return reward

    def step(self, action):
        states = [[] for _ in range(self.simulator.num_flow)]
        end_to_end = []

        def activate_reprofiler(reprofiler, a):
            reprofiler.activate(a)
            # Add a packet forward event if the shaper is turned off.
            if not a:
                for tb in reprofiler.token_buckets:
                    if not tb.idle and tb.head_pointer < len(tb.backlog):
                        tb_packet_number = tb.backlog[tb.head_pointer][1]
                        event = Event(self.time, EventType.FORWARD, reprofiler.flow_idx, tb_packet_number, tb,
                                      flag=False)
                        heapq.heappush(self.simulator.event_pool, event)
            return

        def add_token_to_reprofiler(reprofiler, token_nums):
            # Add token to each token bucket of the reprofiler.
            for tb_idx, token_num in enumerate(token_nums):
                # Remove unused extra token from the previous time step.
                reprofiler.reset_token(self.time, tb_idx)
                # Add extra token for this time step.
                reprofiler.add_token(self.time, tb_idx, token_num)
                # Add a packet forward event if the token bucket has at least one token.
                tb = reprofiler.token_buckets[tb_idx]
                if tb.token >= 1 and len(tb.backlog) > 0:
                    tb_packet_number = tb.backlog[0][1]
                    event = Event(self.time, EventType.FORWARD, reprofiler.flow_idx, tb_packet_number, tb, flag=True)
                    heapq.heappush(self.simulator.event_pool, event)
            return

        def update_wait_time(reprofiler, average_wait_time):
            for tb_idx in range(len(reprofiler.token_buckets)):
                tb = reprofiler.token_buckets[tb_idx]
                tb.average_wait_time = average_wait_time
            return

        # Check the control action type and select the control function.
        if self.action_mode == "add_token":
            if self.passive_tb:
                assert np.issubdtype(action.dtype, np.number)
                action = np.minimum(action, self.max_token_add)
                action_func = add_token_to_reprofiler
            else:
                assert isinstance(action, float)
                action = np.ones((self.simulator.num_flow,), dtype=float) * action
                action_func = update_wait_time
        else:
            assert action.dtype == np.bool_
            action_func = activate_reprofiler
        # Enforce the reprofiling control actions.
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                for flow_idx, a in enumerate(action):
                    action_func(self.simulator.ingress_reprofilers[flow_idx], a)
                    if self.simulator.shaping_mode in ["pfs", "ntb"]:
                        flow_links = self.simulator.flow_path[flow_idx]
                        for link_idx in flow_links:
                            action_func(self.simulator.reprofilers[(link_idx, flow_idx)], a)
                    elif self.simulator.shaping_mode == "ils":
                        flow_links = self.simulator.flow_path[flow_idx]
                        for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                            action_func(self.simulator.reprofilers[(cur_link, next_link)].multi_slope_shapers[flow_idx],
                                        a)
        self.time += self.pause_interval
        # Start the simulation.
        packet_count_old = copy.deepcopy(self.simulator.packet_count)
        self.simulator.simulate()
        # Collect the end-to-end delay experienced by packets that reached the destination during the pause interval.
        for flow_idx, (old_count, new_count) in enumerate(zip(packet_count_old, self.simulator.packet_count)):
            flow_end_to_end = []
            for packet_number in range(old_count, new_count):
                packet_end_to_end = self.simulator.end_to_end_delay[flow_idx][packet_number]
                if packet_end_to_end != -1:
                    packet_end_to_end /= self.simulator.latency_target[flow_idx]
                flow_end_to_end.append(packet_end_to_end)
            end_to_end.append(flow_end_to_end)
        # Record the network status.
        remaining_tokens = []
        for state, tb, p in zip(states, self.simulator.token_buckets, self.simulator.packet_size):
            token_num = tb.peek(self.time)
            remaining_tokens.append(token_num)
            state.append(token_num * p)
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                reprofiler_num = 1 if self.simulator.shaping_mode == "is" else self.simulator.num_link + 1
                reprofiler_backlog = [[0] * reprofiler_num for _ in range(self.simulator.num_flow)]
                for flow_idx, flow_links in enumerate(self.simulator.flow_path):
                    ingress_rb = self.simulator.ingress_reprofilers[flow_idx].peek(self.time)
                    reprofiler_backlog[flow_idx][0] = ingress_rb * self.simulator.packet_size[flow_idx]
                    if self.simulator.shaping_mode == "ils":
                        for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                            rb = self.simulator.reprofilers[(cur_link, next_link)].peek(self.time)
                            reprofiler_backlog[flow_idx][next_link + 1] = rb
                    elif self.simulator.shaping_mode in ["pfs", "ntb"]:
                        for link_idx in flow_links:
                            rb = self.simulator.reprofilers[(link_idx, flow_idx)].peek(self.time)
                            reprofiler_backlog[flow_idx][link_idx + 1] = rb * self.simulator.packet_size[flow_idx]
        scheduler_backlog = [[0] * self.simulator.num_link for _ in range(self.simulator.num_flow)]
        scheduler_utilization = [[0] * self.simulator.num_link for _ in range(self.simulator.num_flow)]
        for flow_idx, flow_links in enumerate(self.simulator.flow_path):
            for link_idx in flow_links:
                sb, su = self.simulator.schedulers[link_idx].peek(self.time)
                scheduler_backlog[flow_idx][link_idx] = sb
                scheduler_utilization[flow_idx][link_idx] = su
        for flow_idx in range(self.simulator.num_flow):
            state = states[flow_idx]
            if self.simulator.scheduling_policy == "fifo":
                if self.simulator.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                    rb = reprofiler_backlog[flow_idx]
                    state.extend(rb)
            sb = scheduler_backlog[flow_idx]
            su = scheduler_utilization[flow_idx]
            state.extend(sb)
            state.extend(su)
        # Update the shaper states.
        if self.action_mode == "add_token" and not self.passive_tb:
            for flow_idx, flow_links in enumerate(self.simulator.flow_path):
                ingress_shaper = self.simulator.ingress_reprofilers[flow_idx]
                for tb in ingress_shaper.token_buckets:
                    tb.update_state(remaining_tokens[flow_idx],
                                    np.array(scheduler_backlog[flow_idx])[flow_links])
        # Compute the reward based on the end-to-end latency and determine whether the episode terminates.
        terminate, exceed_target = True, False
        reward = 0
        for end in end_to_end:
            flow_reward = 0
            for e in end:
                flow_reward += self.reward_function(e)
                if e == -1 or e > 1:
                    exceed_target = True
            if len(self.simulator.event_pool) > 0:
                terminate = False
            # flow_reward = 0 if len(end) == 0 else flow_reward / len(end)
            reward += flow_reward
        # if exceed_target:
        #     reward = self.penalty
        return states, reward, terminate, exceed_target, end_to_end

    def reset(self, arrival_pattern=None):
        self.simulator.reset(arrival_pattern=arrival_pattern)
        self.time = 0
        # Restore the initial event pool if repeating the previous episode.
        if self.repeat and arrival_pattern is None:
            self.simulator.event_pool = copy.deepcopy(self.event_pool_copy)
        else:
            # Add a summary event at each time interval to collect a snapshot of the network.
            for time_step in np.arange(self.pause_interval, self.simulator.simulation_time + self.pause_interval,
                                       self.pause_interval):
                event = Event(time_step, EventType.SUMMARY)
                heapq.heappush(self.simulator.event_pool, event)
        # Set the initial state.
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["pfs", "ils", "ntb"]:
                # Token num, ingress shaper backlog, per-hop shaper backlog, per-hop scheduler backlog and utilization
                state_size = 3 * self.simulator.num_link + 2
            elif self.simulator.shaping_mode == "is":
                # Token num, ingress shaper backlog, per-hop scheduler backlog and utilization
                state_size = 2 * self.simulator.num_link + 2
            elif self.simulator.shaping_mode == "itb":
                # Token num, per-hop scheduler backlog and utilization
                state_size = 2 * self.simulator.num_link + 1
        elif self.simulator.scheduling_policy == "sced":
            # Token num, per-hop scheduler backlog and utilization
            state_size = 2 * self.simulator.num_link + 1
        states = [[0] * state_size for _ in range(self.simulator.num_flow)]
        for state, f, p in zip(states, self.simulator.token_bucket_profile, self.simulator.packet_size):
            state[0] = f[1] * p
        return states
