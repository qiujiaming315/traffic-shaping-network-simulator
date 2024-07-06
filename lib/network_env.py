import heapq
import numpy as np
from dataclasses import dataclass
from enum import Enum

from lib.network_component import NetworkComponent, TokenBucketFluid, TokenBucket, MultiSlopeShaper, FIFOScheduler


class NetworkEnv:
    """A network environment for RL sampling similar to a openai.gym environment."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, interval, terminate_time=1000, pattern="sync",
                 awake_dur=None, high_reward=1, low_reward=0.1, penalty=-10, tor=0.003):
        flow_profile = np.array(flow_profile)
        flow_path = np.array(flow_path)
        reprofiling_delay = np.array(reprofiling_delay)
        self.flow_profile = flow_profile
        self.num_flow = len(flow_profile)
        self.num_link = flow_path.shape[1]
        assert self.num_flow == len(flow_path), "Inconsistent number of flows in flow profile and flow path."
        self.reprofiling_delay = reprofiling_delay
        self.interval = interval
        self.terminate_time = terminate_time
        self.pattern = pattern
        self.awake_dur = terminate_time / 100 if awake_dur is None else awake_dur
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.penalty = penalty
        self.arrival_pattern = self.generate_arrival_pattern()
        # Configure the network components.
        reprofiling_rate = flow_profile[:, 1] / reprofiling_delay
        reprofiling_burst = flow_profile[:, 1] - flow_profile[:, 0] * reprofiling_delay
        self.token_buckets = [TokenBucketFluid(f[0], f[1]) for f in flow_profile]
        self.reprofilers = [[MultiSlopeShaper(flow_idx) for flow_idx in range(self.num_flow)] for _ in
                            range(self.num_link)]
        self.schedulers = []
        self.link_packetization_delay = []
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            link_bandwidth = np.sum(reprofiling_rate[link_flow_mask])
            self.link_packetization_delay.append(np.sum(link_flow_mask) / link_bandwidth)
            self.schedulers.append(FIFOScheduler(link_bandwidth, self.num_flow))
            for flow_idx in np.arange(self.num_flow)[link_flow_mask]:
                self.reprofilers[link_idx][flow_idx] = MultiSlopeShaper(flow_idx,
                                                                        TokenBucket(flow_profile[flow_idx, 0],
                                                                                    reprofiling_burst[flow_idx]),
                                                                        TokenBucket(reprofiling_rate[flow_idx], 0))
        # Compute the expected latency bound (with small tolerance for numerical instability).
        # Packetization delay from reprofilers.
        packetization_delay1_1 = reprofiling_delay / flow_profile[:, 1]
        packetization_delay1_2 = 1 / flow_profile[:, 0]
        packetization_delay1 = np.where(flow_profile[:, 1] >= 1, packetization_delay1_1,
                                        flow_profile[:, 1] * packetization_delay1_1 + (
                                                1 - flow_profile[:, 1]) * packetization_delay1_2)
        packetization_delay1 = np.sum(flow_path > 0, axis=1) * packetization_delay1
        # Packetization delay from schedulers.
        packetization_delay2 = np.sum(np.array(self.link_packetization_delay) * (flow_path > 0), axis=1)
        self.latency_target = (flow_profile[:, 2] + packetization_delay1 + packetization_delay2) * (1 + tor)
        # Connect the network components.
        self.flow_path = []
        for flow_idx, path in enumerate(flow_path):
            flow_links = np.where(path)[0]
            assert len(flow_links) > 0, "Every flow should traverse at least one hop."
            flow_links = flow_links[np.argsort(path[flow_links])]
            self.flow_path.append(flow_links)
            # self.token_buckets[flow_idx].next = self.reprofilers[flow_links[0]][flow_idx]
            # Append an empty component as the terminal.
            self.schedulers[flow_links[-1]].terminal[flow_idx] = True
            self.schedulers[flow_links[-1]].next[flow_idx] = NetworkComponent()
            for link_idx in flow_links:
                self.reprofilers[link_idx][flow_idx].next = self.schedulers[link_idx]
            for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                self.schedulers[cur_link].next[flow_idx] = self.reprofilers[next_link][flow_idx]
        # Set the internal variables.
        self.time = 0
        self.packet_count = [0] * self.num_flow
        self.event_pool = []
        # Add packet arrival events to the event pool.
        self.arrival_time = []
        for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
            flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
            self.arrival_time.append(flow_arrival)
            for arrival in flow_arrival:
                first_reprofiler = self.reprofilers[self.flow_path[flow_idx][0]][flow_idx]
                for tb in first_reprofiler.token_buckets:
                    event = Event(arrival, EventType.ARRIVAL, flow_idx, tb)
                    heapq.heappush(self.event_pool, event)
        self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                               range(self.num_flow)]
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(interval, terminate_time + interval, interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.event_pool, event)
        return

    def reward_function(self, end_to_end_delay, flow_idx):
        # Compute the reward given an end-to-end delay
        reward = 0
        if end_to_end_delay <= self.latency_target[flow_idx]:
            reward = self.low_reward + (1 - end_to_end_delay / self.latency_target[flow_idx]) * (
                    self.high_reward - self.low_reward)
        return reward

    def step(self, action):
        states = [[] for _ in range(self.num_flow)]
        end_to_end = [[] for _ in range(self.num_flow)]
        # Enforce the reprofiling control actions.
        for flow_idx, a in enumerate(action):
            flow_links = self.flow_path[flow_idx]
            for link_idx in flow_links:
                self.reprofilers[link_idx][flow_idx].activate(a, self.time)
                # Add a packet forward event if the shaper is turned off.
                if not a:
                    for tb in self.reprofilers[link_idx][flow_idx].token_buckets:
                        event = Event(self.time, EventType.FORWARD, component=tb)
                        heapq.heappush(self.event_pool, event)
        self.time += self.interval
        while len(self.event_pool) > 0:
            event = heapq.heappop(self.event_pool)
            if event.event_type == EventType.ARRIVAL:
                # Start a busy period by creating a forward event if the component is idle upon arrival.
                if event.component.arrive(event.time, event.flow_idx):
                    forward_event = Event(event.time, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
            elif event.event_type == EventType.FORWARD:
                next_depart, idle, (flow_idx, packet_number, next_component) = event.component.forward(event.time)
                # Submit the next forward event if the component is currently busy.
                if not idle:
                    forward_event = Event(next_depart, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
                # Create a packet arrival event for the next component.
                departed = next_component is not None
                is_internal = isinstance(event.component, TokenBucket)
                is_terminal = isinstance(event.component, FIFOScheduler) and event.component.terminal[flow_idx]
                if departed:
                    if isinstance(next_component, MultiSlopeShaper) and not is_internal:
                        # Create an arrival event for every token bucket from the multi-slope shaper.
                        for tb in next_component.token_buckets:
                            arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, tb)
                            heapq.heappush(self.event_pool, arrival_event)
                    elif not is_terminal:
                        arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, next_component)
                        heapq.heappush(self.event_pool, arrival_event)
                    # Record the packet departure time.
                    if not is_internal:
                        self.departure_time[flow_idx][packet_number - 1].append(event.time)
                    # Update the packet count and compute end-to-end latency.
                    if is_terminal:
                        end_to_end[flow_idx].append(event.time - self.arrival_time[flow_idx][packet_number - 1])
                        self.packet_count[flow_idx] += 1
            elif event.event_type == EventType.SUMMARY:
                # Record the network status.
                for state, tb in zip(states, self.token_buckets):
                    token_num = tb.peek(event.time)
                    state.append(token_num)
                reprofiler_backlog = [[0] * self.num_link for _ in range(self.num_flow)]
                scheduler_backlog = [[0] * self.num_link for _ in range(self.num_flow)]
                for flow_idx, flow_links in enumerate(self.flow_path):
                    for link_idx in flow_links:
                        rb = self.reprofilers[link_idx][flow_idx].peek(event.time)
                        reprofiler_backlog[flow_idx][link_idx] = rb
                        sb = self.schedulers[link_idx].peek(event.time)
                        scheduler_backlog[flow_idx][link_idx] = sb
                for state, rb, sb in zip(states, reprofiler_backlog, scheduler_backlog):
                    state.extend(rb)
                    state.extend(sb)
                break
        # Compute the reward based on the end-to-end latency and determine whether the episode terminates.
        terminate, exceed_target = True, False
        reward = 0
        for flow_idx, end in enumerate(end_to_end):
            flow_reward = 0
            for e in end:
                flow_reward += self.reward_function(e, flow_idx)
                if e > self.latency_target[flow_idx]:
                    exceed_target = True
            if self.packet_count[flow_idx] < len(self.arrival_time[flow_idx]):
                terminate = False
            flow_reward = 0 if len(end) == 0 else flow_reward / len(end)
            reward += flow_reward
        if exceed_target:
            reward = self.penalty
        return states, reward, terminate, exceed_target

    def generate_arrival_pattern(self):
        arrival_pattern = []

        def update_pattern(pattern, arrival_time, arrival_traffic, arrival_rate):
            terminate = arrival_time >= self.terminate_time
            if terminate:
                arrival_traffic -= arrival_rate * (arrival_time - self.terminate_time)
                arrival_time = self.terminate_time
            pattern[0].append(arrival_time)
            pattern[1].append(arrival_traffic)
            return terminate

        if self.pattern.startswith("sync"):
            # Compute the synchronized awake and sleep time for all the flows.
            sync_arrival, sync_time = [0], 0
            awake_bottleneck = np.amax(1 / self.flow_profile[:, 0])
            sleep_bottleneck_smooth = np.amin(self.flow_profile[:, 1] / self.flow_profile[:, 0])
            sleep_bottleneck_burst = np.amax(self.flow_profile[:, 1] / self.flow_profile[:, 0])
            while True:
                # Make sure the awake duration is long enough to allow every flow replenish at least one token.
                while True:
                    awake_dur = np.random.exponential(self.awake_dur)
                    if awake_dur > awake_bottleneck:
                        break
                sync_time += awake_dur
                if sync_time >= self.terminate_time:
                    break
                sync_arrival.append(sync_time)
                if self.pattern == "sync_smooth":
                    sleep_dur = np.random.rand() * sleep_bottleneck_smooth
                else:
                    sleep_dur = sleep_bottleneck_burst
                sync_time += sleep_dur
                if sync_time >= self.terminate_time:
                    break
                sync_arrival.append(sync_time)
            sync_arrival.append(self.terminate_time)
            # Generate the corresponding (synchronized) arrival patterns.
            time_idx, flow_traffic = 0, []
            for flow_idx in range(self.num_flow):
                arrival_pattern.append([[0], [0]])
                flow_traffic.append(self.flow_profile[flow_idx, 1])
            while True:
                for flow_idx in range(self.num_flow):
                    flow_rate = self.flow_profile[flow_idx, 0]
                    flow_burst = self.flow_profile[flow_idx, 1]
                    # Round the cumulative traffic (to ensure packetized arrival).
                    awake_dur = sync_arrival[time_idx + 1] - sync_arrival[time_idx]
                    traffic_new = round(flow_traffic[flow_idx] + awake_dur * flow_rate)
                    awake_new = (traffic_new - flow_traffic[flow_idx]) / flow_rate
                    if time_idx + 2 < len(sync_arrival):
                        sleep_dur = sync_arrival[time_idx + 2] - sync_arrival[time_idx + 1]
                        if awake_new > awake_dur + sleep_dur:
                            # Round down the cumulative traffic instead.
                            traffic_new -= 1
                    else:
                        if awake_new > awake_dur:
                            traffic_new -= 1
                        if traffic_new <= flow_traffic[flow_idx]:
                            continue
                    update_pattern(arrival_pattern[flow_idx], sync_arrival[time_idx], flow_traffic[flow_idx], 0)
                    assert traffic_new > flow_traffic[flow_idx], "Synchronized awake duration too short."
                    awake_new = (traffic_new - flow_traffic[flow_idx]) / flow_rate
                    flow_traffic[flow_idx] = traffic_new
                    update_pattern(arrival_pattern[flow_idx], sync_arrival[time_idx] + awake_new,
                                   flow_traffic[flow_idx], flow_rate)
                    if time_idx + 2 >= len(sync_arrival):
                        continue
                    update_pattern(arrival_pattern[flow_idx], sync_arrival[time_idx + 2], flow_traffic[flow_idx], 0)
                    flow_traffic[flow_idx] += min((sync_arrival[time_idx + 2] - sync_arrival[
                        time_idx] - awake_new) * flow_rate, flow_burst)
                time_idx += 2
                if time_idx >= len(sync_arrival) - 1:
                    break
        else:
            # Generate the (asynchronized) arrival pattern of each flow independently.
            for flow_idx in range(self.num_flow):
                flow_arrival_pattern = [[0], [0]]
                flow_rate = self.flow_profile[flow_idx, 0]
                flow_token = self.flow_profile[flow_idx, 1]
                time, traffic = 0, flow_token
                while True:
                    update_pattern(flow_arrival_pattern, time, traffic, 0)
                    # Make sure the awake duration is long enough.
                    while True:
                        awake_dur = np.random.exponential(self.awake_dur)
                        if round(traffic + awake_dur * flow_rate) > traffic:
                            break
                    # Round the cumulative traffic (to ensure packetized arrival).
                    traffic_new = round(traffic + awake_dur * flow_rate)
                    awake_dur = (traffic_new - traffic) / flow_rate
                    time += awake_dur
                    traffic = traffic_new
                    if update_pattern(flow_arrival_pattern, time, traffic, flow_rate):
                        break
                    sleep_dur = np.random.rand() * (flow_token / flow_rate)
                    time += sleep_dur
                    if update_pattern(flow_arrival_pattern, time, traffic, 0):
                        break
                    traffic += sleep_dur * flow_rate
                arrival_pattern.append(flow_arrival_pattern)
        return arrival_pattern

    def reset(self):
        self.arrival_pattern = self.generate_arrival_pattern()
        for token_bucket in self.token_buckets:
            token_bucket.reset()
        for link_reprofiler, scheduler in zip(self.reprofilers, self.schedulers):
            for reprofiler in link_reprofiler:
                reprofiler.reset()
            scheduler.reset()
        self.time = 0
        self.packet_count = [0] * len(self.arrival_pattern)
        self.event_pool = []
        # Add packet arrival events to the event pool.
        self.arrival_time = []
        for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
            flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
            self.arrival_time.append(flow_arrival)
            for arrival in flow_arrival:
                first_reprofiler = self.reprofilers[self.flow_path[flow_idx][0]][flow_idx]
                for tb in first_reprofiler.token_buckets:
                    event = Event(arrival, EventType.ARRIVAL, flow_idx, tb)
                    heapq.heappush(self.event_pool, event)
        self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                               range(self.num_flow)]
        # Add a summary event at each time interval to collect a snapshot of the network.
        for time_step in np.arange(self.interval, self.terminate_time + self.interval, self.interval):
            event = Event(time_step, EventType.SUMMARY)
            heapq.heappush(self.event_pool, event)
        # Set the initial state.
        states = [[0] * (2 * self.num_link + 1) for _ in range(self.num_flow)]
        for state, f in zip(states, self.flow_profile):
            state[0] = f[1]
        return states


class EventType(Enum):
    SUMMARY = 1
    FORWARD = 2
    ARRIVAL = 3


@dataclass
class Event:
    time: float
    event_type: EventType
    flow_idx: int = 0
    component: NetworkComponent = NetworkComponent()

    def __lt__(self, other):
        return (self.time, self.event_type.value) < (other.time, other.event_type.value)
