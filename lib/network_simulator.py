import copy
import heapq
import numbers
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from lib.traffic_shapers import NetworkComponent, TokenBucketFluid, PassiveExtraTokenBucket, \
    DeSyncExtraTokenBucket, MultiSlopeShaper, InterleavedShaper
from lib.schedulers import Scheduler, FIFOScheduler, SCEDScheduler, TokenBucketSCED, MultiSlopeShaperSCED


class NetworkSimulator:
    """A network simulator supporting several types of traffic shaping and scheduling policy."""

    def __init__(self, flow_profile, flow_path, reprofiling_delay, simulation_time=1000, scheduling_policy="fifo",
                 shaping_mode="pfs", buffer_bound="infinite", traffic_cycle_period=5.0, clock_drift_std=0.01,
                 load_perturbation=(0.05, 0.005), reboot_inter_arrival_avg=200.0, reboot_time_avg=5.0,
                 arrival_pattern=None, passive_tb=True, keep_per_hop_departure=True, repeat=False, scaling_factor=1.0,
                 packet_size=1, propagation_delay=0, tor=0.003, seed=None):
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
        assert traffic_cycle_period > 0, "Please set the traffic arrival cycle to be a positive value."
        self.traffic_cycle_period = traffic_cycle_period
        assert clock_drift_std >= 0, "Please set the standard deviation of the clock drift be a non-negative value."
        self.clock_drift_std = clock_drift_std
        self.load_perturbation = np.array(load_perturbation)
        assert len(load_perturbation) == 2 and np.all(self.load_perturbation >= 0), "Please set the average and " \
                                                                                    "standard deviation of the load " \
                                                                                    "perturbation to non-negative " \
                                                                                    "values."
        assert reboot_inter_arrival_avg > 0, "Please set the inter-arrival time of reboot events to a positive value."
        self.reboot_inter_arrival_avg = reboot_inter_arrival_avg
        assert reboot_time_avg > 0, "Please set the average reboot time to a positive value."
        self.reboot_time_avg = reboot_time_avg
        assert reboot_inter_arrival_avg >= 10 * reboot_time_avg, "Please make sure the average inter-arrival time of " \
                                                                 "reboot events is at least 10 times larger than the " \
                                                                 "average reboot time."
        if not passive_tb:
            assert shaping_mode == "is", "The shaping mode must be ingress shaping (is) to support shapers with " \
                                         "desynchronized extra tokens."
        self.passive_tb = passive_tb
        self.keep_per_hop_departure = keep_per_hop_departure
        self.repeat = repeat
        self.scaling_factor = scaling_factor
        if isinstance(packet_size, Iterable):
            assert len(packet_size) == self.num_flow, "Please set the packet size either as a single value, " \
                                                      "or as a list of values, one for each flow."
        else:
            packet_size = [packet_size] * self.num_flow
        self.packet_size = np.array(packet_size)
        if isinstance(propagation_delay, Iterable):
            assert len(propagation_delay) == self.num_link, "Please set the packet propapation delay either as a " \
                                                            "single value, or as a list of values, one for each link."
        else:
            propagation_delay = [propagation_delay] * self.num_link
        self.propagation_delay = np.array(propagation_delay)
        # Initiate the random number generator.
        self.rng = np.random.default_rng(seed)
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

        def get_reprofiler(flow_idx, latency_min):
            if self.shaping_mode == "ntb":
                if self.passive_tb:
                    tb = PassiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                 self.token_bucket_profile[flow_idx, 1])
                else:
                    tb = DeSyncExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                self.token_bucket_profile[flow_idx, 1],
                                                self.latency_target[flow_idx], latency_min,
                                                self.token_bucket_profile[flow_idx, 0])
                return MultiSlopeShaper(flow_idx, tb)
            else:
                if self.passive_tb:
                    tb1 = PassiveExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                  token_bucket_reprofiling_burst[flow_idx])
                    tb2 = PassiveExtraTokenBucket(token_bucket_reprofiling_rate[flow_idx], 1)
                else:
                    tb1 = DeSyncExtraTokenBucket(self.token_bucket_profile[flow_idx, 0],
                                                 token_bucket_reprofiling_burst[flow_idx],
                                                 self.latency_target[flow_idx], latency_min,
                                                 self.token_bucket_profile[flow_idx, 0])
                    tb2 = DeSyncExtraTokenBucket(token_bucket_reprofiling_rate[flow_idx], 1,
                                                 self.latency_target[flow_idx], latency_min,
                                                 self.token_bucket_profile[flow_idx, 0])
                return MultiSlopeShaper(flow_idx, tb1, tb2)

        # Register the token buckets profile of each flow.
        self.token_buckets = [TokenBucketFluid(f[0], f[1]) for f in self.token_bucket_profile]
        # Register the shapers according to the traffic shaping mode (under FIFO scheduling).
        latency_min = np.amin(self.latency_target)
        if scheduling_policy == "fifo":
            if shaping_mode in ["pfs", "ils", "is", "ntb"]:
                self.ingress_reprofilers = []
                for flow_idx in range(self.num_flow):
                    ingress_reprofiler = get_reprofiler(flow_idx, latency_min)
                    ingress_reprofiler.ingress = True
                    self.ingress_reprofilers.append(ingress_reprofiler)
            if shaping_mode in ["pfs", "ntb"]:
                self.reprofilers = dict()
                for link_idx in range(self.num_link):
                    link_flow_mask = flow_path[:, link_idx] > 0
                    for flow_idx in np.arange(self.num_flow)[link_flow_mask]:
                        self.reprofilers[(link_idx, flow_idx)] = get_reprofiler(flow_idx, latency_min)
            elif shaping_mode == "ils":
                self.reprofilers = dict()
                flow_group = defaultdict(list)
                for flow_idx, flow_links in enumerate(self.flow_path):
                    for cur_link, next_link in zip(flow_links[:-1], flow_links[1:]):
                        flow_group[(cur_link, next_link)].append(flow_idx)
                for link_pair in flow_group.keys():
                    self.reprofilers[link_pair] = InterleavedShaper(self.packet_size,
                                                                    *[get_reprofiler(flow_idx, latency_min) for flow_idx
                                                                      in flow_group[link_pair]])
        # Register the link schedulers.
        self.schedulers = []
        for link_idx in range(self.num_link):
            link_flow_mask = flow_path[:, link_idx] > 0
            if scheduling_policy == "fifo":
                self.schedulers.append(FIFOScheduler(self.link_bandwidth[link_idx], self.packet_size,
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
        self.packet_count_terminal = [0] * self.num_flow
        self.packet_count_shaper = [0] * self.num_flow
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
        self.shaping_delay = [[-1] * len(self.arrival_time[flow_idx]) for flow_idx in range(self.num_flow)]
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
                assert isinstance(event.component, DeSyncExtraTokenBucket)
                granted = event.component.get_extra_tokens(event.time, event.packet_number)
                # Add a packet forward event if at least one extra token is granted.
                if granted:
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
                if isinstance(event.component, DeSyncExtraTokenBucket):
                    scheduled_event_number, scheduled_wait_time = event.component.schedule_extra_tokens(event.time)
                    # Grant extra tokens after a random wait time.
                    if scheduled_wait_time != -1:
                        extra_token_event = Event(event.time + scheduled_wait_time, EventType.EXTRA_TOKEN,
                                                  flow_idx=event.flow_idx, packet_number=scheduled_event_number,
                                                  component=event.component)
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
                is_ingress = isinstance(event.component, MultiSlopeShaper) and event.component.ingress
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
                    # Compute the shaping delay.
                    if is_ingress:
                        self.packet_count_shaper[forwarded_idx] = forwarded_number + 1
                        self.shaping_delay[forwarded_idx][forwarded_number] = event.time - \
                                                                              self.arrival_time[forwarded_idx][
                                                                                  forwarded_number]
                    # Update the packet count and compute end-to-end latency.
                    if is_terminal:
                        self.packet_count_terminal[forwarded_idx] = forwarded_number + 1
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
        """Generate the periodic arrival patterns with clock drifts and load perturbations."""
        arrival_pattern = []
        for flow_idx in range(self.num_flow):
            arrival_pattern.append([[0], [0]])

        def update_pattern(pattern, arrival_time, arrival_traffic, arrival_rate):
            terminate = arrival_time >= self.simulation_time
            if terminate:
                arrival_traffic -= arrival_rate * (arrival_time - self.simulation_time)
                arrival_time = self.simulation_time
            pattern[0].append(arrival_time)
            pattern[1].append(arrival_traffic)
            return terminate

        # sleep_bottleneck_min = np.amin(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
        # print(f"{sleep_bottleneck_min:.3f}")
        # sleep_bottleneck_max = np.amax(self.token_bucket_profile[:, 1] / self.token_bucket_profile[:, 0])
        # print(f"{sleep_bottleneck_max:.3f}")
        # Randomly set the clock drift of each traffic source.
        clock_drifts = self.rng.normal(loc=0, scale=self.clock_drift_std, size=(self.num_flow,))
        # Randomly set system reboot events.
        reboot_events, last_reboot = [], 0
        while last_reboot < self.simulation_time:
            reboot_events.append(last_reboot)
            reboot_inter_arrival = self.rng.exponential(self.reboot_inter_arrival_avg)
            last_reboot += reboot_inter_arrival
        # Randomly set the reboot time of each reboot event.
        reboot_times = [0]
        for reboot_idx in range(1, len(reboot_events)):
            reboot_time = self.rng.exponential(self.reboot_time_avg)
            if reboot_idx != len(reboot_events) - 1:
                # Ensure every system reboot happens before the next reboot.
                while reboot_events[reboot_idx] + reboot_time >= reboot_events[reboot_idx + 1]:
                    reboot_time = self.rng.exponential(self.reboot_time_avg)
            reboot_times.append(reboot_time)
        # Initiate the traffic arrival states of the flows.
        flow_traffic, flow_token, flow_last_start, flow_cycle, flow_reboot = [], [], [], [], []
        for flow_idx in range(self.num_flow):
            flow_traffic.append(0)
            flow_token.append(self.token_bucket_profile[flow_idx, 1])
            flow_last_start.append(0)
            flow_cycle.append(0)
            flow_reboot.append(0)
        while True:
            terminate = True
            for flow_idx in range(self.num_flow):
                flow_rate = self.token_bucket_profile[flow_idx, 0]
                flow_burst = self.token_bucket_profile[flow_idx, 1]
                flow_standard_start = flow_cycle[flow_idx] * self.traffic_cycle_period
                flow_reboot_idx = flow_reboot[flow_idx]
                flow_last_reboot = reboot_events[flow_reboot_idx] + reboot_times[flow_reboot_idx]
                cumulative_drift = flow_cycle[flow_idx] * clock_drifts[flow_idx]
                flow_load_perturbation = self.rng.normal(loc=self.load_perturbation[0],
                                                         scale=self.load_perturbation[1])
                flow_start_time = flow_standard_start + flow_last_reboot + cumulative_drift + flow_load_perturbation
                # Ensure the start time is not negative.
                while flow_start_time < 0:
                    # Reset the clock drift if the drift is a large negative value.
                    if cumulative_drift < 0 and flow_reboot_idx == 0 and flow_cycle[flow_idx] == 1:
                        clock_drifts[flow_idx] = self.rng.normal(loc=0, scale=self.clock_drift_std)
                        cumulative_drift = flow_cycle[flow_idx] * clock_drifts[flow_idx]
                    # Reset the load perturbation value.
                    else:
                        flow_load_perturbation = self.rng.normal(loc=self.load_perturbation[0],
                                                                 scale=self.load_perturbation[1])
                    flow_start_time = flow_standard_start + flow_last_reboot + cumulative_drift + flow_load_perturbation
                # Wait for reboot if a reboot event happens.
                next_reboot_idx = flow_reboot_idx + 1
                if next_reboot_idx < len(reboot_events) and flow_start_time > reboot_events[next_reboot_idx]:
                    flow_reboot[flow_idx] += 1
                    flow_cycle[flow_idx] = 0
                    terminate = False
                    continue
                # Skip if the next burst starts after the simulation finishes.
                if flow_start_time >= self.simulation_time:
                    continue
                terminate = False
                # Update arrival pattern after a silent period.
                flow_token[flow_idx] += (flow_start_time - flow_last_start[flow_idx]) * flow_rate
                flow_token[flow_idx] = min(flow_token[flow_idx], flow_burst)
                update_pattern(arrival_pattern[flow_idx], flow_start_time, flow_traffic[flow_idx], 0)
                flow_traffic_burst = flow_traffic[flow_idx] + flow_token[flow_idx]
                # Update arrival pattern after an awake period (including an initial burst).
                # Round the cumulative traffic to the nearest integer (to ensure packetized arrival).
                traffic_new = round(flow_traffic_burst)
                awake_dur = max((traffic_new - flow_traffic_burst) / flow_rate, 0)
                period_duration = (self.traffic_cycle_period + clock_drifts[flow_idx]) * (
                        flow_cycle[flow_idx] + 1) + flow_last_reboot - flow_start_time
                if awake_dur > period_duration:
                    # If the awake duration is too large, round down the cumulative traffic instead.
                    traffic_new -= 1
                    awake_dur = max((traffic_new - flow_traffic_burst) / flow_rate, 0)
                if traffic_new > flow_traffic_burst:
                    update_pattern(arrival_pattern[flow_idx], flow_start_time, flow_traffic_burst, 0)
                    flow_token[flow_idx] = 0
                else:
                    flow_token[flow_idx] -= traffic_new - flow_traffic[flow_idx]
                flow_traffic[flow_idx] = traffic_new
                update_pattern(arrival_pattern[flow_idx], flow_start_time + awake_dur, traffic_new, flow_rate)
                flow_last_start[flow_idx] = flow_start_time + awake_dur
                flow_cycle[flow_idx] += 1
            if terminate:
                break
        return arrival_pattern

    def reset(self, seed=None, arrival_pattern=None):
        self.rng = np.random.default_rng(seed)
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
        self.packet_count_terminal = [0] * self.num_flow
        self.packet_count_shaper = [0] * self.num_flow
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
        self.shaping_delay = [[-1] * len(self.arrival_time[flow_idx]) for flow_idx in range(self.num_flow)]
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
