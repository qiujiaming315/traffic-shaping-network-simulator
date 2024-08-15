import heapq
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from lib.network_component import NetworkComponent, TokenBucketFluid, TokenBucket, MultiSlopeShaper, InterleavedShaper, \
    FIFOScheduler


class NetworkSimulator:
    """A network simulator supporting several types of traffic shaping and scheduling policy."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, shaping_mode="per_flow",
                 arrival_pattern_type="sync_burst", awake_dur=None, arrival_pattern=None, keep_per_hop_departure=True,
                 scaling_factor=1.0, packet_size=1, tor=0.003):
        flow_profile = np.array(flow_profile)
        flow_path = np.array(flow_path)
        reprofiling_delay = np.array(reprofiling_delay)
        self.flow_profile = flow_profile
        self.num_flow = len(flow_profile)
        self.num_link = flow_path.shape[1]
        assert self.num_flow == len(flow_path), "Inconsistent number of flows in flow profile and flow path."
        self.reprofiling_delay = reprofiling_delay
        self.simulation_time = simulation_time
        valid_mode = shaping_mode in ["per_flow", "interleaved", "ingress", "none"]
        assert valid_mode, "Please choose a shaping mode among 'per_flow', 'interleaved', 'ingress', 'none'."
        self.shaping_mode = shaping_mode
        valid_pattern = arrival_pattern_type in ["sync_burst", "sync_smooth", "async"] or arrival_pattern is not None
        assert valid_pattern, "Please choose an arrival pattern type among 'sync_burst', 'sync_smooth', and 'async'."
        self.arrival_pattern_type = arrival_pattern_type
        self.awake_dur = simulation_time / 100 if awake_dur is None else awake_dur
        self.keep_per_hop_departure = keep_per_hop_departure
        self.scaling_factor = scaling_factor
        if isinstance(packet_size, Iterable):
            assert len(packet_size) == self.num_flow, "Please set the packet size either as a single value, " \
                                                      "or as a list of values, one for each flow."
        else:
            packet_size = [packet_size] * self.num_flow
        self.packet_size = np.array(packet_size)
        # Compute the token bucket profile of each flow using 1 packet size as the unit.
        self.token_bucket_profile = self.flow_profile[:, :2] / self.packet_size[:, np.newaxis]
        self.token_bucket_profile[:, 1] += 1
        self.arrival_pattern = self.generate_arrival_pattern() if arrival_pattern is None else arrival_pattern
        # Configure the network components.
        reprofiling_rate = flow_profile[:, 1] / reprofiling_delay
        reprofiling_burst = flow_profile[:, 1] - flow_profile[:, 0] * reprofiling_delay
        token_bucket_reprofiling_rate = reprofiling_rate / self.packet_size
        token_bucket_reprofiling_burst = reprofiling_burst / self.packet_size + 1

        def get_reprofiler(flow_idx):
            return MultiSlopeShaper(flow_idx, TokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                          token_bucket_reprofiling_burst[flow_idx]),
                                    TokenBucket(token_bucket_reprofiling_rate[flow_idx], 1))

        # Retrieve the path of each flow.
        self.flow_path = []
        for flow_idx, path in enumerate(flow_path):
            flow_links = np.where(path)[0]
            assert len(flow_links) > 0, "Every flow should traverse at least one hop."
            flow_links = flow_links[np.argsort(path[flow_links])]
            self.flow_path.append(flow_links)
        # Register the token buckets profile of each flow.
        self.token_buckets = [TokenBucketFluid(f[0], f[1]) for f in self.token_bucket_profile]
        # Register the shapers according to the traffic shaping mode.
        if shaping_mode == "per_flow":
            self.reprofilers = dict()
            for link_idx in range(self.num_link):
                link_flow_mask = flow_path[:, link_idx] > 0
                for flow_idx in np.arange(self.num_flow)[link_flow_mask]:
                    self.reprofilers[(link_idx, flow_idx)] = get_reprofiler(flow_idx)
        elif shaping_mode == "interleaved":
            self.ingress_reprofilers = [get_reprofiler(flow_idx) for flow_idx in range(self.num_flow)]
            self.reprofilers = dict()
            flow_group = defaultdict(list)
            for flow_idx, flow_links in enumerate(self.flow_path):
                for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                    flow_group[(cur_link, next_link)].append(flow_idx)
            for link_pair in flow_group.keys():
                self.reprofilers[link_pair] = InterleavedShaper(self.packet_size,
                                                                *[get_reprofiler(flow_idx) for flow_idx in
                                                                  flow_group[link_pair]])
        elif shaping_mode == "ingress":
            self.reprofilers = [get_reprofiler(flow_idx) for flow_idx in range(self.num_flow)]
        # Register the link schedulers.
        self.schedulers = []
        self.link_packetization_delay = np.zeros((self.num_link,), dtype=float)
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            link_bandwidth = np.sum(reprofiling_rate[link_flow_mask]) * scaling_factor
            if np.sum(link_flow_mask) > 0:
                self.link_packetization_delay[link_idx] = np.sum(self.packet_size[link_flow_mask]) / link_bandwidth
            self.schedulers.append(FIFOScheduler(link_bandwidth, self.packet_size))
        # Compute the expected latency bound (with small tolerance for numerical instability).
        # Compute packetization delay from schedulers.
        packetization_delay = np.sum(self.link_packetization_delay * (flow_path > 0), axis=1)
        self.latency_target = (flow_profile[:, 2] + packetization_delay) * (1 + tor)
        # self.latency_target = (reprofiling_delay + packetization_delay) * (1 + tor)
        # Connect the network components.
        for flow_idx, flow_links in enumerate(self.flow_path):
            # Append an empty component to the terminal component.
            self.schedulers[flow_links[-1]].terminal[flow_idx] = True
            self.schedulers[flow_links[-1]].next[flow_idx] = NetworkComponent()
            # Connect the shapers and the schedulers according to the network topology.
            if self.shaping_mode == "per_flow":
                for link_idx in flow_links:
                    self.reprofilers[(link_idx, flow_idx)].next = self.schedulers[link_idx]
                for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                    self.schedulers[cur_link].next[flow_idx] = self.reprofilers[(next_link, flow_idx)]
            elif self.shaping_mode == "interleaved":
                self.ingress_reprofilers[flow_idx].next = self.schedulers[flow_links[0]]
                for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                    self.schedulers[cur_link].next[flow_idx] = self.reprofilers[(cur_link, next_link)]
                    self.reprofilers[(cur_link, next_link)].next = self.schedulers[next_link]
            elif self.shaping_mode in ["ingress", "none"]:
                if self.shaping_mode == "ingress":
                    self.reprofilers[flow_idx].next = self.schedulers[flow_links[0]]
                for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                    self.schedulers[cur_link].next[flow_idx] = self.schedulers[next_link]
        # Set the internal variables.
        self.scheduler_max_backlog = [0] * self.num_link
        self.event_pool = []
        # Add packet arrival events to the event pool.
        self.arrival_time = []
        for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
            flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
            self.arrival_time.append(flow_arrival)
            for packet_number, arrival in enumerate(flow_arrival):
                if shaping_mode in ["per_flow", "interleaved", "ingress"]:
                    if shaping_mode == "per_flow":
                        first_reprofiler = self.reprofilers[(self.flow_path[flow_idx][0], flow_idx)]
                    elif shaping_mode == "interleaved":
                        first_reprofiler = self.ingress_reprofilers[flow_idx]
                    else:
                        first_reprofiler = self.reprofilers[flow_idx]
                    for tb in first_reprofiler.token_buckets:
                        event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, tb)
                        heapq.heappush(self.event_pool, event)
                if shaping_mode == "none":
                    first_scheduler = self.schedulers[self.flow_path[flow_idx][0]]
                    event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, first_scheduler)
                    heapq.heappush(self.event_pool, event)
        # Keep track of the packet departure time at every hop if specified.
        if keep_per_hop_departure:
            self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                                   range(self.num_flow)]
        else:
            self.departure_time = None
        self.end_to_end_delay = [[-1] * len(self.arrival_time[flow_idx]) for flow_idx in range(self.num_flow)]
        return

    def simulate(self):
        while len(self.event_pool) > 0:
            event = heapq.heappop(self.event_pool)
            if event.event_type == EventType.ARRIVAL:
                # Start a busy period by creating a forward event if the component is idle upon arrival.
                if event.component.arrive(event.time, event.packet_number, event.flow_idx, event.internal):
                    forward_event = Event(event.time, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
            elif event.event_type == EventType.FORWARD:
                next_depart, idle, flow_idx, packet_number, next_component = event.component.forward(event.time)
                # Submit the next forward event if the component is currently busy.
                if not idle:
                    forward_event = Event(next_depart, EventType.FORWARD, component=event.component)
                    heapq.heappush(self.event_pool, forward_event)
                # Create a packet arrival event for the next component.
                departed = next_component is not None
                is_next_ms = isinstance(next_component, MultiSlopeShaper)
                is_internal_tb = isinstance(event.component, TokenBucket) and event.component.internal
                is_next_interleaved = isinstance(next_component, InterleavedShaper)
                is_internal_ms = isinstance(event.component, MultiSlopeShaper) and event.component.internal
                is_terminal = isinstance(event.component, FIFOScheduler) and event.component.terminal[flow_idx]
                if departed:
                    ms_arrival = is_next_ms and not is_internal_tb
                    interleaved_arrival = is_next_interleaved and not is_internal_ms
                    if not (is_terminal or ms_arrival):
                        arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, packet_number, next_component,
                                              internal=is_internal_ms)
                        heapq.heappush(self.event_pool, arrival_event)
                    if ms_arrival or interleaved_arrival:
                        ms_component = next_component if ms_arrival else next_component.multi_slope_shapers[flow_idx]
                        # Create an arrival event for every token bucket of the multi-slope shaper.
                        for tb in ms_component.token_buckets:
                            arrival_event = Event(event.time, EventType.ARRIVAL, flow_idx, packet_number, tb)
                            heapq.heappush(self.event_pool, arrival_event)
                    # Record the packet departure time.
                    if not (is_internal_tb or is_internal_ms) and self.keep_per_hop_departure:
                        self.departure_time[flow_idx][packet_number].append(event.time)
                    # Update the packet count and compute end-to-end latency.
                    if is_terminal:
                        self.end_to_end_delay[flow_idx][packet_number] = event.time - self.arrival_time[flow_idx][
                            packet_number]
            elif event.event_type == EventType.SUMMARY:
                break
        # Track the maximum backlog size at each link scheduler.
        for link_idx, scheduler in enumerate(self.schedulers):
            self.scheduler_max_backlog[link_idx] = scheduler.max_backlog_size
        return

    def generate_arrival_pattern(self):
        arrival_pattern = []

        # Set a large enough awake duration.
        awake_bottleneck = np.amax(1 / self.token_bucket_profile[:, 0])
        self.awake_dur = max(self.awake_dur, 1.1 * awake_bottleneck)

        def update_pattern(pattern, arrival_time, arrival_traffic, arrival_rate):
            terminate = arrival_time >= self.simulation_time
            if terminate:
                arrival_traffic -= arrival_rate * (arrival_time - self.simulation_time)
                arrival_time = self.simulation_time
            pattern[0].append(arrival_time)
            pattern[1].append(arrival_traffic)
            return terminate

        if self.arrival_pattern_type.startswith("sync"):
            # Compute the synchronized awake and sleep time for all the flows.
            sync_arrival, sync_time = [0], 0
            awake_bottleneck = np.amax(1 / self.token_bucket_profile[:, 0])
            sleep_bottleneck_smooth = np.amin(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
            sleep_bottleneck_burst = np.amax(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
            while True:
                # Make sure the awake duration is long enough to allow every flow replenish at least one token.
                while True:
                    awake_dur = np.random.exponential(self.awake_dur)
                    if awake_dur > awake_bottleneck:
                        break
                sync_time += awake_dur
                if sync_time >= self.simulation_time:
                    break
                sync_arrival.append(sync_time)
                if self.arrival_pattern_type == "sync_smooth":
                    sleep_dur = np.random.rand() * sleep_bottleneck_smooth
                else:
                    sleep_dur = sleep_bottleneck_burst
                sync_time += sleep_dur
                if sync_time >= self.simulation_time:
                    break
                sync_arrival.append(sync_time)
            sync_arrival.append(self.simulation_time)
            # Generate the corresponding (synchronized) arrival patterns.
            time_idx, flow_traffic = 0, []
            for flow_idx in range(self.num_flow):
                arrival_pattern.append([[0], [0]])
                flow_traffic.append(self.token_bucket_profile[flow_idx, 1])
            while True:
                for flow_idx in range(self.num_flow):
                    flow_rate = self.token_bucket_profile[flow_idx, 0]
                    flow_burst = self.token_bucket_profile[flow_idx, 1]
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
                flow_rate = self.token_bucket_profile[flow_idx, 0]
                flow_token = self.token_bucket_profile[flow_idx, 1]
                time, traffic = 0, 0
                while True:
                    sleep_dur = np.random.rand() * (flow_token / flow_rate)
                    time += sleep_dur
                    if update_pattern(flow_arrival_pattern, time, traffic, 0):
                        break
                    traffic += sleep_dur * flow_rate
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
                arrival_pattern.append(flow_arrival_pattern)
        return arrival_pattern

    def reset(self, arrival_pattern=None):
        self.arrival_pattern = self.generate_arrival_pattern() if arrival_pattern is None else arrival_pattern
        for token_bucket in self.token_buckets:
            token_bucket.reset()
        for scheduler in self.schedulers:
            scheduler.reset()
        if self.shaping_mode in ["per_flow", "interleaved"]:
            for key in self.reprofilers:
                self.reprofilers[key].reset()
        if self.shaping_mode == "interleaved":
            for reprofiler in self.ingress_reprofilers:
                reprofiler.reset()
        if self.shaping_mode == "ingress":
            for reprofiler in self.reprofilers:
                reprofiler.reset()
        self.scheduler_max_backlog = [0] * self.num_link
        self.event_pool = []
        # Add packet arrival events to the event pool.
        self.arrival_time = []
        for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
            flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
            self.arrival_time.append(flow_arrival)
            for packet_number, arrival in enumerate(flow_arrival):
                if self.shaping_mode in ["per_flow", "interleaved", "ingress"]:
                    if self.shaping_mode == "per_flow":
                        first_reprofiler = self.reprofilers[(self.flow_path[flow_idx][0], flow_idx)]
                    elif self.shaping_mode == "interleaved":
                        first_reprofiler = self.ingress_reprofilers[flow_idx]
                    else:
                        first_reprofiler = self.reprofilers[flow_idx]
                    for tb in first_reprofiler.token_buckets:
                        event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, tb)
                        heapq.heappush(self.event_pool, event)
                if self.shaping_mode == "none":
                    first_scheduler = self.schedulers[self.flow_path[flow_idx][0]]
                    event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, first_scheduler)
                    heapq.heappush(self.event_pool, event)
        if self.keep_per_hop_departure:
            self.departure_time = [[[] for _ in range(len(self.arrival_time[flow_idx]))] for flow_idx in
                                   range(self.num_flow)]
        else:
            self.departure_time = None
        self.end_to_end_delay = [[-1] * len(self.arrival_time[flow_idx]) for flow_idx in range(self.num_flow)]
        return


class EventType(Enum):
    SUMMARY = 1
    FORWARD = 2
    ARRIVAL = 3


@dataclass
class Event:
    time: float
    event_type: EventType
    flow_idx: int = 0
    packet_number: int = 0
    component: NetworkComponent = NetworkComponent()
    internal: bool = False

    def __lt__(self, other):
        return (self.time, self.event_type.value, self.internal) < (other.time, other.event_type.value, other.internal)
