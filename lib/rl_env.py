import copy
import heapq
import numpy as np

from lib.network_simulator import NetworkSimulator, Event, EventType


class RLNetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, scheduling_policy="fifo",
                 shaping_mode="per_flow", buffer_bound="infinite", arrival_pattern_type="sync_burst", awake_dur=None,
                 awake_dist="exponential", sync_jitter=0, arrival_pattern=None, keep_per_hop_departure=True,
                 scaling_factor=1.0, packet_size=1, tor=0.003, pause_interval=1, high_reward=1, low_reward=0.1,
                 penalty=-10):
        self.simulator = NetworkSimulator(flow_profile, flow_path, reprofiling_delay, simulation_time=simulation_time,
                                          scheduling_policy=scheduling_policy, shaping_mode=shaping_mode,
                                          buffer_bound=buffer_bound, arrival_pattern_type=arrival_pattern_type,
                                          awake_dur=awake_dur, awake_dist=awake_dist, sync_jitter=sync_jitter,
                                          arrival_pattern=arrival_pattern,
                                          keep_per_hop_departure=keep_per_hop_departure, scaling_factor=scaling_factor,
                                          packet_size=packet_size, tor=tor)
        self.pause_interval = pause_interval
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        self.time = 0
        self.num_action = self.simulator.num_flow
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(pause_interval, simulation_time + pause_interval, pause_interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.simulator.event_pool, event)
        return

    def reward_function(self, end_to_end_delay, flow_idx):
        # Compute the reward given an end-to-end delay
        reward = 0
        if end_to_end_delay != -1 and end_to_end_delay <= self.simulator.latency_target[flow_idx]:
            reward = self.low_reward + (1 - end_to_end_delay / self.simulator.latency_target[flow_idx]) * (
                    self.high_reward - self.low_reward)
        return reward

    def step(self, action):
        states = [[] for _ in range(self.simulator.num_flow)]
        end_to_end = []

        def activate_reprofiler(reprofiler, a):
            reprofiler.activate(a, self.time)
            # Add a packet forward event if the shaper is turned off.
            if not a:
                for tb in reprofiler.token_buckets:
                    event = Event(self.time, EventType.FORWARD, component=tb)
                    heapq.heappush(self.simulator.event_pool, event)
            return

        # Enforce the reprofiling control actions.
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["per_flow", "interleaved", "ingress"]:
                for flow_idx, a in enumerate(action):
                    if self.simulator.shaping_mode == "per_flow":
                        flow_links = self.simulator.flow_path[flow_idx]
                        for link_idx in flow_links:
                            activate_reprofiler(self.simulator.reprofilers[(link_idx, flow_idx)], a)
                    elif self.simulator.shaping_mode == "interleaved":
                        activate_reprofiler(self.simulator.ingress_reprofilers[flow_idx], a)
                        flow_links = self.simulator.flow_path[flow_idx]
                        for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                            activate_reprofiler(
                                self.simulator.reprofilers[(cur_link, next_link)].multi_slope_shapers[flow_idx],
                                a)
                    else:
                        activate_reprofiler(self.simulator.reprofilers[flow_idx], a)
        self.time += self.pause_interval
        # Start the simulation.
        packet_count_old = copy.deepcopy(self.simulator.packet_count)
        self.simulator.simulate()
        # Collect the end-to-end delay experienced by packets that reached the destination during the pause interval.
        for flow_idx, (old_count, new_count) in enumerate(zip(packet_count_old, self.simulator.packet_count)):
            flow_end_to_end = []
            for packet_number in range(old_count, new_count):
                flow_end_to_end.append(self.simulator.end_to_end_delay[flow_idx][packet_number])
            end_to_end.append(flow_end_to_end)
        # Record the network status.
        for state, tb, p in zip(states, self.simulator.token_buckets, self.simulator.packet_size):
            token_num = tb.peek(self.time)
            state.append(token_num * p)
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["per_flow", "interleaved"]:
                reprofiler_backlog = [[0] * self.simulator.num_link for _ in range(self.simulator.num_flow)]
            elif self.simulator.shaping_mode == "ingress":
                reprofiler_backlog = [0] * self.simulator.num_flow
            for flow_idx, flow_links in enumerate(self.simulator.flow_path):
                if self.simulator.shaping_mode == "ingress":
                    rb = self.simulator.reprofilers[flow_idx].peek(self.time)
                    reprofiler_backlog[flow_idx] = rb * self.simulator.packet_size[flow_idx]
                if self.simulator.shaping_mode == "interleaved":
                    ingress_rb = self.simulator.ingress_reprofilers[flow_idx].peek(self.time)
                    reprofiler_backlog[flow_idx][0] = ingress_rb * self.simulator.packet_size[flow_idx]
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        rb = self.simulator.reprofilers[(cur_link, next_link)].peek(self.time)
                        reprofiler_backlog[flow_idx][next_link] = rb
                for link_idx in flow_links:
                    if self.simulator.shaping_mode == "per_flow":
                        rb = self.simulator.reprofilers[(link_idx, flow_idx)].peek(self.time)
                        reprofiler_backlog[flow_idx][link_idx] = rb * self.simulator.packet_size[flow_idx]
        scheduler_backlog = [[0] * self.simulator.num_link for _ in range(self.simulator.num_flow)]
        for flow_idx, flow_links in enumerate(self.simulator.flow_path):
            for link_idx in flow_links:
                sb = self.simulator.schedulers[link_idx].peek(self.time)
                scheduler_backlog[flow_idx][link_idx] = sb
        for flow_idx in range(self.simulator.num_flow):
            state = states[flow_idx]
            if self.simulator.scheduling_policy == "fifo":
                if self.simulator.shaping_mode in ["per_flow", "interleaved", "ingress"]:
                    rb = reprofiler_backlog[flow_idx]
                    if self.simulator.shaping_mode in ["per_flow", "interleaved"]:
                        state.extend(rb)
                    else:
                        state.append(rb)
            sb = scheduler_backlog[flow_idx]
            state.extend(sb)
        # Compute the reward based on the end-to-end latency and determine whether the episode terminates.
        terminate, exceed_target = True, False
        reward = 0
        for flow_idx, end in enumerate(end_to_end):
            flow_reward = 0
            for e in end:
                flow_reward += self.reward_function(e, flow_idx)
                if e == -1 or e > self.simulator.latency_target[flow_idx]:
                    exceed_target = True
            if len(self.simulator.event_pool) > 0:
                terminate = False
            flow_reward = 0 if len(end) == 0 else flow_reward / len(end)
            reward += flow_reward
        if exceed_target:
            reward = self.penalty
        return states, reward, terminate, exceed_target

    def reset(self, arrival_pattern=None):
        self.simulator.reset(arrival_pattern=arrival_pattern)
        self.time = 0
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(self.pause_interval, self.simulator.simulation_time + self.pause_interval,
                                   self.pause_interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.simulator.event_pool, event)
        # Set the initial state.
        if self.simulator.scheduling_policy == "fifo":
            if self.simulator.shaping_mode in ["per_flow", "interleaved"]:
                state_size = 2 * self.simulator.num_link + 1
            elif self.simulator.shaping_mode == "ingress":
                state_size = self.simulator.num_link + 2
            elif self.simulator.shaping_mode == "none":
                state_size = self.simulator.num_link + 1
        elif self.simulator.scheduling_policy == "sced":
            state_size = self.simulator.num_link + 1
        states = [[0] * state_size for _ in range(self.simulator.num_flow)]
        for state, f, p in zip(states, self.simulator.token_bucket_profile, self.simulator.packet_size):
            state[0] = f[1] * p
        return states
