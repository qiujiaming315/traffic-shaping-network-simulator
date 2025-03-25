import copy
import heapq
import numbers
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from lib.traffic_shapers import NetworkComponent, TokenBucketFluid, PassiveExtraTokenBucket, \
    ProactiveExtraTokenBucket, MultiSlopeShaper, InterleavedShaper
from lib.schedulers import Scheduler, FIFOScheduler, SCEDScheduler, TokenBucketSCED, MultiSlopeShaperSCED


class NetworkSimulator:
    """A network simulator supporting several types of traffic shaping and scheduling policy."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, scheduling_policy="fifo",
                 shaping_mode="pfs", buffer_bound="infinite", arrival_pattern_type="sync", sync_jitter=0,
                 periodic_arrival_ratio=1.0, periodic_pattern_weight=(0.8, 0.1, 0.1), awake_dur=0,
                 awake_dist="constant", sleep_dur="max", sleep_dist="constant", arrival_pattern=None, passive_tb=True,
                 tb_average_wait_time=0.5, keep_per_hop_departure=True, repeat=False, scaling_factor=1.0, packet_size=1,
                 busy_period_window_size=0, propagation_delay=0, tor=0.003):
        flow_profile = np.array(flow_profile)
        flow_path = np.array(flow_path)
        reprofiling_delay = np.array(reprofiling_delay)
        self.flow_profile = flow_profile
        self.num_flow = len(flow_profile)
        self.num_link = flow_path.shape[1]
        assert self.num_flow == len(flow_path), "Inconsistent number of flows in flow profile and flow path."
        self.reprofiling_delay = reprofiling_delay
        self.simulation_time = simulation_time
        valid_scheduling = scheduling_policy in ["fifo", "sced"]
        assert valid_scheduling, "Please choose a scheduling policy between 'fifo' and 'sced'."
        self.scheduling_policy = scheduling_policy
        valid_shaping = shaping_mode in ["pfs", "ils", "is", "ntb", "itb"]
        assert valid_shaping, "Please choose a shaping mode among 'pfs', 'ils', 'is', 'ntb' and 'itb'."
        self.shaping_mode = shaping_mode
        valid_buffer = buffer_bound in ["infinite", "with_shaping"]
        assert valid_buffer, "Please choose a buffer bound between 'infinite' and 'with_shaping'."
        self.buffer_bound = buffer_bound
        valid_pattern = arrival_pattern_type in ["sync", "async"] or arrival_pattern is not None
        assert valid_pattern, "Please choose an arrival pattern type between 'sync' and 'async'."
        self.arrival_pattern_type = arrival_pattern_type
        self.sync_jitter = sync_jitter
        assert 0 <= periodic_arrival_ratio <= 1, "Please choose a periodic arrival ratio within the range [0, 1]."
        self.periodic_arrival_ratio = periodic_arrival_ratio
        assert isinstance(periodic_pattern_weight, tuple) and len(periodic_pattern_weight) == 3, \
            "Please set the flow periodic arrival pattern weights to be a tuple of 3 elements representing the " \
            "weight of selecting 'awake', 'sleep', or 'keep awake' as the periodic pattern, respectively."
        self.periodic_pattern_weight = np.array(periodic_pattern_weight)
        assert np.all(self.periodic_pattern_weight > 0), "Please set the flow awake sampling weights to positive " \
                                                         "values."
        self.periodic_pattern_weight /= np.sum(self.periodic_pattern_weight)
        valid_awake = awake_dist in ["exponential", "constant"]
        assert valid_awake, "Please choose an awake duration distribution between 'exponential' and 'constant'."
        self.awake_dist = awake_dist
        self.awake_dur = awake_dur
        if self.awake_dist == "exponential":
            assert self.awake_dur > 0, "Please set a positive awake duration if the distribution is exponential."
        valid_sleep = sleep_dist in ["uniform", "constant"]
        assert valid_sleep, "Please choose a sleep duration distribution between 'uniform' and 'constant'."
        self.sleep_dist = sleep_dist
        if sleep_dur not in ["min", "max"]:
            try:
                sleep_dur = float(sleep_dur)
            except ValueError:
                print("Please set sleep duration to 'min', 'max', or a number.")
        self.sleep_dur = sleep_dur
        if not passive_tb:
            assert shaping_mode == "is", "The shaping mode must be ingress shaping (is) to support shapers with " \
                                         "proactively granted extra tokens."
        self.passive_tb = passive_tb
        assert tb_average_wait_time > 0, "Please set a positive average wait time to grant extra tokens."
        self.tb_average_wait_time = tb_average_wait_time
        self.keep_per_hop_departure = keep_per_hop_departure
        self.repeat = repeat
        self.scaling_factor = scaling_factor
        if isinstance(packet_size, Iterable):
            assert len(packet_size) == self.num_flow, "Please set the packet size either as a single value, " \
                                                      "or as a list of values, one for each flow."
        else:
            packet_size = [packet_size] * self.num_flow
        self.packet_size = np.array(packet_size)
        self.busy_period_window_size = busy_period_window_size
        if isinstance(propagation_delay, Iterable):
            assert len(propagation_delay) == self.num_link, "Please set the packet propapation delay either as a " \
                                                            "single value, or as a list of values, one for each link."
        else:
            propagation_delay = [propagation_delay] * self.num_link
        self.propagation_delay = np.array(propagation_delay)
        # Compute the token bucket profile of each flow using 1 packet size as the unit.
        self.token_bucket_profile = self.flow_profile[:, :2] / self.packet_size[:, np.newaxis]
        self.token_bucket_profile[:, 1] += 1
        self.arrival_pattern = self.generate_arrival_pattern() if arrival_pattern is None else arrival_pattern
        # Configure the network components.
        reprofiling_rate = flow_profile[:, 1] / reprofiling_delay
        reprofiling_burst = flow_profile[:, 1] - flow_profile[:, 0] * reprofiling_delay
        token_bucket_reprofiling_rate = reprofiling_rate / self.packet_size
        token_bucket_reprofiling_burst = reprofiling_burst / self.packet_size + 1
        remaining_delay = (flow_profile[:, 2] - reprofiling_delay) / np.sum(flow_path > 0, axis=1)
        # Retrieve the path of each flow.
        self.flow_path = []
        for flow_idx, path in enumerate(flow_path):
            flow_links = np.where(path)[0]
            assert len(flow_links) > 0, "Every flow should traverse at least one hop."
            flow_links = flow_links[np.argsort(path[flow_links])]
            self.flow_path.append(flow_links)
        # Compute the link parameters.
        self.link_bandwidth = np.zeros((self.num_link,), dtype=float)
        self.link_packetization_delay = np.zeros((self.num_link,), dtype=float)
        self.link_buffer = [None] * self.num_link
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            link_bandwidth = np.sum(reprofiling_rate[link_flow_mask]) * scaling_factor
            if np.sum(link_flow_mask) > 0:
                self.link_bandwidth[link_idx] = link_bandwidth
                self.link_packetization_delay[link_idx] = np.sum(self.packet_size[link_flow_mask]) / link_bandwidth
                if buffer_bound == "with_shaping":
                    # Compute buffer bound assuming FIFO scheduling and per-flow shaping.
                    # Add a maximum packet size to avoid buffer overflow due to numerical issues.
                    self.link_buffer[link_idx] = (np.sum(self.packet_size[link_flow_mask]) + np.amax(
                        self.packet_size[link_flow_mask])) * (1 + tor)
        # Compute the expected latency bound (with small tolerance for numerical instability).
        # Compute packetization delay from schedulers.
        packetization_delay = np.sum(self.link_packetization_delay * (flow_path > 0), axis=1)
        total_propagation_delay = np.sum(self.propagation_delay * (flow_path > 0), axis=1)
        # self.latency_target = (reprofiling_delay + packetization_delay + total_propagation_delay) * (1 + tor)
        self.latency_target = (flow_profile[:, 2] + packetization_delay + total_propagation_delay) * (1 + tor)

        def get_reprofiler(flow_idx):
            flow_transmission_delay = 1 / self.link_bandwidth[self.flow_path[flow_idx]]
            flow_propagation_delay = self.propagation_delay[self.flow_path[flow_idx]]
            if self.shaping_mode == "ntb":
                if self.passive_tb:
                    tb = PassiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                 self.token_bucket_profile[flow_idx, 1])
                else:
                    tb = ProactiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                   self.token_bucket_profile[flow_idx, 1], self.tb_average_wait_time,
                                                   self.packet_size[flow_idx], self.latency_target[flow_idx],
                                                   flow_transmission_delay, flow_propagation_delay)
                return MultiSlopeShaper(flow_idx, tb)
            else:
                if self.passive_tb:
                    tb1 = PassiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                  token_bucket_reprofiling_burst[flow_idx])
                    tb2 = PassiveExtraTokenBucket(token_bucket_reprofiling_rate[flow_idx], 1)
                else:
                    tb1 = ProactiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                    token_bucket_reprofiling_burst[flow_idx], self.tb_average_wait_time,
                                                    self.packet_size[flow_idx], self.latency_target[flow_idx],
                                                    flow_transmission_delay, flow_propagation_delay)
                    tb2 = ProactiveExtraTokenBucket(token_bucket_reprofiling_rate[flow_idx], 1,
                                                    self.tb_average_wait_time, self.packet_size[flow_idx],
                                                    self.latency_target[flow_idx], flow_transmission_delay,
                                                    flow_propagation_delay)
                return MultiSlopeShaper(flow_idx, tb1, tb2)

        # Register the token buckets profile of each flow.
        self.token_buckets = [TokenBucketFluid(f[0], f[1]) for f in self.token_bucket_profile]
        # Register the shapers according to the traffic shaping mode (under FIFO scheduling).
        if scheduling_policy == "fifo":
            if shaping_mode in ["pfs", "ils", "is", "ntb"]:
                self.ingress_reprofilers = [get_reprofiler(flow_idx) for flow_idx in range(self.num_flow)]
            if shaping_mode in ["pfs", "ntb"]:
                self.reprofilers = dict()
                for link_idx in range(self.num_link):
                    link_flow_mask = flow_path[:, link_idx] > 0
                    for flow_idx in np.arange(self.num_flow)[link_flow_mask]:
                        self.reprofilers[(link_idx, flow_idx)] = get_reprofiler(flow_idx)
            elif shaping_mode == "ils":
                self.reprofilers = dict()
                flow_group = defaultdict(list)
                for flow_idx, flow_links in enumerate(self.flow_path):
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        flow_group[(cur_link, next_link)].append(flow_idx)
                for link_pair in flow_group.keys():
                    self.reprofilers[link_pair] = InterleavedShaper(self.packet_size,
                                                                    *[get_reprofiler(flow_idx) for flow_idx in
                                                                      flow_group[link_pair]])
        # Register the link schedulers.
        self.schedulers = []
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            if scheduling_policy == "fifo":
                self.schedulers.append(FIFOScheduler(self.link_bandwidth[link_idx], self.packet_size,
                                                     busy_period_window_size=self.busy_period_window_size,
                                                     propagation_delay=self.propagation_delay[link_idx],
                                                     buffer_size=self.link_buffer[link_idx]))
            elif scheduling_policy == "sced":
                def get_reprofiler_sced(flow_idx):
                    return MultiSlopeShaperSCED(TokenBucketSCED(self.token_bucket_profile[flow_idx, 0],
                                                                token_bucket_reprofiling_burst[flow_idx]),
                                                TokenBucketSCED(token_bucket_reprofiling_rate[flow_idx], 1))

                self.schedulers.append(
                    SCEDScheduler(self.link_bandwidth[link_idx], self.packet_size, remaining_delay,
                                  *[(flow_idx, get_reprofiler_sced(flow_idx)) for flow_idx in
                                    np.arange(self.num_flow)[link_flow_mask]],
                                  busy_period_window_size=self.busy_period_window_size,
                                  propagation_delay=self.propagation_delay[link_idx],
                                  buffer_size=self.link_buffer[link_idx]))
        # Connect the network components.
        for flow_idx, flow_links in enumerate(self.flow_path):
            # Append an empty component to the terminal component.
            self.schedulers[flow_links[-1]].terminal[flow_idx] = True
            self.schedulers[flow_links[-1]].next[flow_idx] = NetworkComponent()
            # Connect the shapers and the schedulers according to the network topology.
            if self.scheduling_policy == "fifo":
                if self.shaping_mode in ["pfs", "ntb"]:
                    self.ingress_reprofilers[flow_idx].next = self.reprofilers[(flow_links[0], flow_idx)]
                    for link_idx in flow_links:
                        self.reprofilers[(link_idx, flow_idx)].next = self.schedulers[link_idx]
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        self.schedulers[cur_link].next[flow_idx] = self.reprofilers[(next_link, flow_idx)]
                elif self.shaping_mode == "ils":
                    self.ingress_reprofilers[flow_idx].next = self.schedulers[flow_links[0]]
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        self.schedulers[cur_link].next[flow_idx] = self.reprofilers[(cur_link, next_link)]
                        self.reprofilers[(cur_link, next_link)].next = self.schedulers[next_link]
                elif self.shaping_mode == "is":
                    self.ingress_reprofilers[flow_idx].next = self.schedulers[flow_links[0]]
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        self.schedulers[cur_link].next[flow_idx] = self.schedulers[next_link]
                elif self.shaping_mode == "itb":
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        self.schedulers[cur_link].next[flow_idx] = self.schedulers[next_link]
            elif self.scheduling_policy == "sced":
                for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                    self.schedulers[cur_link].next[flow_idx] = self.schedulers[next_link]
        # Set the internal variables.
        self.packet_count = [0] * self.num_flow
        self.scheduler_max_backlog = [0] * self.num_link
        if scheduling_policy == "fifo":
            if shaping_mode in ["pfs", "ils", "is", "ntb"]:
                self.ingress_reprofiler_max_backlog = [0] * len(self.ingress_reprofilers)
            if shaping_mode in ["pfs", "ils", "ntb"]:
                self.reprofiler_max_backlog = [0] * len(self.reprofilers)
        self.event_pool = []
        # Add packet arrival events to the event pool.
        self.arrival_time = []
        for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
            flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
            self.arrival_time.append(flow_arrival)
            for packet_number, arrival in enumerate(flow_arrival):
                if scheduling_policy == "fifo" and shaping_mode in ["pfs", "ils", "is", "ntb"]:
                    first_reprofiler = self.ingress_reprofilers[flow_idx]
                    for tb in first_reprofiler.token_buckets:
                        event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, tb)
                        heapq.heappush(self.event_pool, event)
                else:
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
        # Keep the initial event pool for restoration upon resetting if repeatable.
        self.event_pool_copy = None
        if self.repeat:
            self.event_pool_copy = copy.deepcopy(self.event_pool)
        return

    def simulate(self):
        while len(self.event_pool) > 0:
            event = heapq.heappop(self.event_pool)
            if event.event_type == EventType.EXTRA_TOKEN:
                event.component.clear_backlog(event.time)
                # Add a packet forward event if at least one extra token is granted.
                if len(event.component.backlog) > 0:
                    tb_packet_number = event.component.backlog[0][1]
                    forward_event = Event(event.time, EventType.FORWARD, event.flow_idx, tb_packet_number,
                                          event.component, flag=True)
                    heapq.heappush(self.event_pool, forward_event)
            elif event.event_type == EventType.ARRIVAL:
                # Start a busy period by creating a forward event if the component is idle upon arrival.
                if event.component.arrive(event.time, event.packet_number, event.flow_idx, event.flag):
                    forward_event = Event(event.time, EventType.FORWARD, event.flow_idx, event.packet_number,
                                          event.component, flag=True)
                    heapq.heappush(self.event_pool, forward_event)
                # Add a non-conformant packet forward event if the token bucket is not active.
                if isinstance(event.component, PassiveExtraTokenBucket) and not event.component.active:
                    forward_event = Event(event.time, EventType.FORWARD, event.flow_idx, event.packet_number,
                                          event.component, flag=False)
                    heapq.heappush(self.event_pool, forward_event)
                # Grant extra tokens after a random wait time.
                if isinstance(event.component, ProactiveExtraTokenBucket) and event.component.extra_eligible:
                    extra_token_event = Event(event.time + event.component.wait_time, EventType.EXTRA_TOKEN,
                                              flow_idx=event.flow_idx, component=event.component)
                    heapq.heappush(self.event_pool, extra_token_event)
            elif event.event_type == EventType.FORWARD:
                (next_depart, next_idx, next_number, idle, forwarded_idx, forwarded_number,
                 next_component) = event.component.forward(event.time, event.packet_number, event.flow_idx, event.flag)
                # Submit the next forward event if the component is currently busy.
                if not idle:
                    forward_event = Event(next_depart, EventType.FORWARD, next_idx, next_number, event.component,
                                          flag=event.flag)
                    heapq.heappush(self.event_pool, forward_event)
                # Create a packet arrival event for the next component.
                departed = next_component is not None
                is_next_ms = isinstance(next_component, MultiSlopeShaper)
                is_internal_tb = isinstance(event.component, PassiveExtraTokenBucket) and event.component.internal
                is_next_interleaved = isinstance(next_component, InterleavedShaper)
                is_internal_ms = isinstance(event.component, MultiSlopeShaper) and event.component.internal
                is_terminal = isinstance(event.component, Scheduler) and event.component.terminal[forwarded_idx]
                link_propagation_delay = event.component.propagation_delay if isinstance(event.component,
                                                                                         Scheduler) else 0
                if departed:
                    ms_arrival = is_next_ms and not is_internal_tb
                    interleaved_arrival = is_next_interleaved and not is_internal_ms
                    if not (is_terminal or ms_arrival):
                        arrival_event = Event(event.time + link_propagation_delay, EventType.ARRIVAL, forwarded_idx,
                                              forwarded_number, next_component, flag=is_internal_ms)
                        heapq.heappush(self.event_pool, arrival_event)
                    if ms_arrival or interleaved_arrival:
                        ms_component = next_component if ms_arrival else next_component.multi_slope_shapers[
                            forwarded_idx]
                        # Create an arrival event for every token bucket of the multi-slope shaper.
                        for tb in ms_component.token_buckets:
                            arrival_event = Event(event.time + link_propagation_delay, EventType.ARRIVAL, forwarded_idx,
                                                  forwarded_number, tb)
                            heapq.heappush(self.event_pool, arrival_event)
                    # Record the packet departure time.
                    if not (is_internal_tb or is_internal_ms) and self.keep_per_hop_departure:
                        self.departure_time[forwarded_idx][forwarded_number].append(event.time)
                    # Update the packet count and compute end-to-end latency.
                    if is_terminal:
                        self.packet_count[forwarded_idx] = forwarded_number + 1
                        self.end_to_end_delay[forwarded_idx][forwarded_number] = event.time - \
                                                                                 self.arrival_time[forwarded_idx][
                                                                                     forwarded_number]
            elif event.event_type == EventType.SUMMARY:
                break
        # Track the maximum backlog size at each link scheduler.
        for link_idx, scheduler in enumerate(self.schedulers):
            self.scheduler_max_backlog[link_idx] = scheduler.max_backlog_size
        # Track the maximum backlog size at each reprofiler.
        if self.scheduling_policy == "fifo":
            if self.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                for flow_idx in range(self.num_flow):
                    self.ingress_reprofiler_max_backlog[flow_idx] = max(
                        tb.max_backlog_size for tb in self.ingress_reprofilers[flow_idx].token_buckets) * \
                                                                    self.packet_size[flow_idx]
            if self.shaping_mode in ["pfs", "ntb"]:
                for idx, key in enumerate(self.reprofilers.keys()):
                    flow_idx = key[1]
                    self.reprofiler_max_backlog[idx] = max(
                        tb.max_backlog_size for tb in self.reprofilers[key].token_buckets) * self.packet_size[flow_idx]
            elif self.shaping_mode == "ils":
                for idx, key in enumerate(self.reprofilers.keys()):
                    self.reprofiler_max_backlog[idx] = self.reprofilers[key].max_backlog_size
        return

    def generate_arrival_pattern(self):
        arrival_pattern = []
        for flow_idx in range(self.num_flow):
            arrival_pattern.append([[0], [0]])
        # Select flows with periodic arrival patterns.
        periodic_flow = np.random.choice(self.num_flow, int(self.num_flow * self.periodic_arrival_ratio), replace=False)
        # Set the arrival pattern of non-periodic flows.
        non_periodic_mask = np.ones((self.num_flow,), dtype=bool)
        non_periodic_mask[periodic_flow] = False
        non_periodic_flow = np.arange(self.num_flow)[non_periodic_mask]
        for flow_idx in non_periodic_flow:
            # Set non-periodic flows to send the maximum amount of traffic throughout the simulation.
            flow_rate = self.token_bucket_profile[flow_idx, 0]
            flow_burst = self.token_bucket_profile[flow_idx, 1]
            arrival_pattern[flow_idx][0].append(0)
            arrival_pattern[flow_idx][1].append(flow_burst)
            arrival_pattern[flow_idx][0].append(self.simulation_time)
            arrival_pattern[flow_idx][1].append(flow_burst + self.simulation_time * flow_rate)

        def update_pattern(pattern, arrival_time, arrival_traffic, arrival_rate):
            terminate = arrival_time >= self.simulation_time
            if terminate:
                arrival_traffic -= arrival_rate * (arrival_time - self.simulation_time)
                arrival_time = self.simulation_time
            pattern[0].append(arrival_time)
            pattern[1].append(arrival_traffic)
            return terminate

        if self.arrival_pattern_type == "sync":
            # Compute the synchronized awake and sleep time for all the flows.
            sync_arrival, sync_time = [0], 0
            sleep_bottleneck_min = np.amin(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
            sleep_bottleneck_max = np.amax(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
            while True:
                # Set the awake duration of all the flows.
                if self.awake_dist == "exponential":
                    awake_dur = np.random.exponential(self.awake_dur)
                else:
                    awake_dur = self.awake_dur
                sync_time += awake_dur
                if sync_time >= self.simulation_time:
                    break
                sync_arrival.append(sync_time)
                # Set the sleep duration of all the flows (including a synchronization jitter).
                if self.sleep_dur == "min":
                    sleep_dur = sleep_bottleneck_min
                elif self.sleep_dur == "max":
                    sleep_dur = sleep_bottleneck_max
                else:
                    sleep_dur = self.sleep_dur
                if self.sleep_dist == "uniform":
                    sleep_dur = np.random.rand() * sleep_dur
                sync_time += sleep_dur + self.sync_jitter
                if sync_time >= self.simulation_time:
                    break
                sync_arrival.append(sync_time)
            sync_arrival.append(self.simulation_time)
            # Generate the corresponding (synchronized) arrival patterns.
            time_idx, flow_traffic, flow_token, flow_current_pattern = 0, [], [], []
            flow_awake, flow_traffic_burst, flow_sleep = [], [], []
            for i in range(len(periodic_flow)):
                flow_traffic.append(0)
                flow_token.append(self.token_bucket_profile[periodic_flow[i], 1])
                flow_current_pattern.append(0)
                flow_awake.append(-1)
                flow_traffic_burst.append(-1)
                flow_sleep.append(0)
            while True:
                for i in range(len(periodic_flow)):
                    flow_idx = periodic_flow[i]
                    flow_rate = self.token_bucket_profile[flow_idx, 0]
                    flow_burst = self.token_bucket_profile[flow_idx, 1]
                    period_end_idx = time_idx + 2 if time_idx + 2 < len(sync_arrival) else time_idx + 1
                    current_pattern = flow_current_pattern[i]
                    if current_pattern != 1:
                        # If the current pattern is not sleep, the next pattern can be any one of the three.
                        next_pattern = np.random.choice(3, 1, p=self.periodic_pattern_weight)[0]
                    else:
                        # Otherwise, the next pattern must be either awake or sleep.
                        next_pattern = np.random.choice(2, 1, p=self.periodic_pattern_weight[:2] / np.sum(
                            self.periodic_pattern_weight[:2]))[0]
                    # Update the flow arrival pattern.
                    flow_current_pattern[i] = next_pattern
                    if current_pattern == 1:
                        # Let the flow sleep through the current period.
                        flow_sleep[i] += sync_arrival[period_end_idx] - sync_arrival[time_idx]
                    else:
                        flow_jitter = np.random.rand() * self.sync_jitter
                        if current_pattern == 0:
                            # Skip if the next burst starts after the simulation finishes.
                            if sync_arrival[time_idx] + flow_jitter >= self.simulation_time:
                                continue
                            # Update arrival pattern after a sleep period.
                            flow_token[i] += (flow_sleep[i] + flow_jitter) * flow_rate
                            flow_token[i] = min(flow_token[i], flow_burst)
                            update_pattern(arrival_pattern[flow_idx], sync_arrival[time_idx] + flow_jitter,
                                           flow_traffic[i], 0)
                            # Keep track of the flow bursting event.
                            flow_awake[i] = sync_arrival[time_idx] + flow_jitter
                            flow_traffic_burst[i] = flow_traffic[i] + flow_token[i]
                        if next_pattern != 2 or time_idx + 2 >= len(sync_arrival) - 1:
                            # Update arrival pattern after an awake period (including an initial burst).
                            # Round the cumulative traffic (to ensure packetized arrival).
                            awake_dur = sync_arrival[time_idx + 1] + flow_jitter - flow_awake[i]
                            traffic_new = round(flow_traffic_burst[i] + awake_dur * flow_rate)
                            awake_dur = max((traffic_new - flow_traffic_burst[i]) / flow_rate, 0)
                            period_duration = sync_arrival[period_end_idx] - flow_awake[i]
                            if awake_dur > period_duration:
                                # If the awake duration is too large, round down the cumulative traffic instead.
                                traffic_new -= 1
                                awake_dur = max((traffic_new - flow_traffic_burst[i]) / flow_rate, 0)
                            if traffic_new > flow_traffic_burst[i]:
                                update_pattern(arrival_pattern[flow_idx], flow_awake[i],
                                               flow_traffic_burst[i], 0)
                                flow_token[i] = 0
                            else:
                                flow_token[i] -= traffic_new - flow_traffic[i]
                            flow_traffic[i] = traffic_new
                            update_pattern(arrival_pattern[flow_idx], flow_awake[i] + awake_dur,
                                           traffic_new, flow_rate)
                            if time_idx + 2 >= len(sync_arrival):
                                continue
                            # Keep track of the flow sleep duration.
                            sleep_dur = period_duration - awake_dur
                            flow_sleep[i] = sleep_dur
                time_idx += 2
                if time_idx >= len(sync_arrival) - 1:
                    break
        else:
            # Generate the (asynchronized) arrival pattern of each flow independently.
            for flow_idx in periodic_flow:
                flow_rate = self.token_bucket_profile[flow_idx, 0]
                flow_token = self.token_bucket_profile[flow_idx, 1]
                time, traffic, token, sleep_overtime, awake_overtime = 0, 0, flow_token, 0, 0
                current_pattern, traffic_burst = 0, -1
                while True:
                    if current_pattern != 1:
                        # If the current pattern is not sleep, the next pattern can be any one of the three.
                        next_pattern = np.random.choice(3, 1, p=self.periodic_pattern_weight)[0]
                    else:
                        # Otherwise, the next pattern must be either awake or sleep.
                        next_pattern = np.random.choice(2, 1, p=self.periodic_pattern_weight[:2] / np.sum(
                            self.periodic_pattern_weight[:2]))[0]
                    # Compute the sleep and awake time for this period.
                    if self.sleep_dur in ["min", "max"]:
                        sleep_dur = (flow_token / flow_rate)
                    else:
                        sleep_dur = self.sleep_dur
                    if self.sleep_dist == "uniform":
                        sleep_dur = np.random.rand() * sleep_dur
                    if self.awake_dist == "exponential":
                        awake_dur = np.random.exponential(self.awake_dur)
                    else:
                        awake_dur = self.awake_dur
                    if current_pattern == 1:
                        # Let the flow sleep through the current period.
                        sleep_overtime += sleep_dur + awake_dur
                    else:
                        if current_pattern == 0:
                            # Update arrival pattern after a sleep period.
                            time += sleep_dur + sleep_overtime
                            token += (sleep_dur + sleep_overtime) * flow_rate
                            token = min(token, flow_token)
                            sleep_overtime = 0
                            if update_pattern(arrival_pattern[flow_idx], time, traffic, 0):
                                break
                            traffic_burst = traffic + token
                        else:
                            awake_overtime += sleep_dur
                        if next_pattern != 2:
                            # Update arrival pattern after an awake period (including an initial burst).
                            # Round the cumulative traffic (to ensure packetized arrival).
                            traffic_new = round(traffic_burst + (awake_dur + awake_overtime) * flow_rate)
                            awake_overtime = 0
                            if traffic_new > traffic_burst:
                                update_pattern(arrival_pattern[flow_idx], time, traffic_burst, 0)
                                awake_dur = (traffic_new - traffic_burst) / flow_rate
                                time += awake_dur
                                token = 0
                            else:
                                token -= traffic_new - traffic
                            traffic = traffic_new
                            if update_pattern(arrival_pattern[flow_idx], time, traffic, flow_rate):
                                break
                        else:
                            awake_overtime += awake_dur
                    current_pattern = next_pattern
        return arrival_pattern

    def reset(self, arrival_pattern=None):
        if arrival_pattern is not None:
            self.arrival_pattern = arrival_pattern
        elif not self.repeat:
            self.arrival_pattern = self.generate_arrival_pattern()
        for scheduler in self.schedulers:
            scheduler.reset()
        if self.scheduling_policy == "fifo":
            if self.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                for reprofiler in self.ingress_reprofilers:
                    reprofiler.reset()
            if self.shaping_mode in ["pfs", "ils", "ntb"]:
                for key in self.reprofilers:
                    self.reprofilers[key].reset()
        self.packet_count = [0] * self.num_flow
        self.scheduler_max_backlog = [0] * self.num_link
        if self.scheduling_policy == "fifo":
            if self.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                self.ingress_reprofiler_max_backlog = [0] * len(self.ingress_reprofilers)
            if self.shaping_mode in ["pfs", "ils", "ntb"]:
                self.reprofiler_max_backlog = [0] * len(self.reprofilers)
        self.event_pool = []
        # Restore the initial event pool if repeating the previous simulation.
        if self.repeat and arrival_pattern is None:
            self.event_pool = copy.deepcopy(self.event_pool_copy)
        else:
            # Reset the token bucket states.
            for token_bucket in self.token_buckets:
                token_bucket.reset()
            # Add packet arrival events to the event pool.
            self.arrival_time = []
            for flow_idx, fluid_arrival in enumerate(self.arrival_pattern):
                flow_arrival = self.token_buckets[flow_idx].forward(fluid_arrival)
                self.arrival_time.append(flow_arrival)
                for packet_number, arrival in enumerate(flow_arrival):
                    if self.scheduling_policy == "fifo" and self.shaping_mode in ["pfs", "ils", "is", "ntb"]:
                        first_reprofiler = self.ingress_reprofilers[flow_idx]
                        for tb in first_reprofiler.token_buckets:
                            event = Event(arrival, EventType.ARRIVAL, flow_idx, packet_number, tb)
                            heapq.heappush(self.event_pool, event)
                    else:
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
    EXTRA_TOKEN = 4


@dataclass
class Event:
    time: float
    event_type: EventType
    flow_idx: int = 0
    packet_number: int = 0
    component: NetworkComponent = NetworkComponent()
    flag: bool = False

    def __lt__(self, other):
        return (self.time, self.event_type.value, self.flag, self.packet_number) < (
            other.time, other.event_type.value, other.flag, other.packet_number)
