import bisect
import numpy as np


class NetworkComponent:

    def __init__(self):
        self.next = None
        self.idle = True
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        """Method to add an arriving packet to backlog."""
        return

    def forward(self, time, packet_number, component_idx, is_conformant):
        """Method to release a packet from the backlog."""
        return

    def peek(self, time):
        """Method to check the status of the network component."""
        return

    def reset(self):
        self.idle = True
        return


class TokenBucket(NetworkComponent):

    def __init__(self, rate, burst, component_idx=0, internal=False):
        self.rate = rate
        self.burst = burst
        self.component_idx = component_idx
        self.internal = internal
        self.buffer_size = burst
        self.active = True
        self.backlog = []
        self.max_backlog_size = 0
        self.head_pointer = 0
        self.token = burst
        self.depart = 0
        super().__init__()
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        self.backlog.append((time, packet_number))
        self.max_backlog_size = max(self.max_backlog_size, len(self.backlog) - self.head_pointer)
        return self.idle

    def forward(self, time, packet_number, component_idx, is_conformant):
        if is_conformant:
            # When the packet forwarding is conformant with the token bucket profile.
            # Check if the packet to forward has the right sequence number.
            if len(self.backlog) == 0 or self.backlog[0][1] != packet_number:
                # Redundant forward event. Ignore.
                return time, 0, 0, True, 0, 0, None
            # Release the forwarded packet.
            forwarded_number, forwarded_idx, next_component = 0, 0, None
            if self.idle:
                # Initiate a busy period.
                token, _ = self.peek(time)
                self.token = token
                self.depart = time
                self.idle = False
            else:
                # Release the forwarded packet.
                _, forwarded_number = self.backlog.pop(0)
                # A packet arrives at the next component if the actual head packet is released.
                if self.head_pointer == 0:
                    forwarded_idx, next_component = self.component_idx, self.next
                # Update the head packet pointer.
                self.head_pointer = max(self.head_pointer - 1, 0)
                # The token number increment should not be bounded by the buffer size during a busy period.
                # This ensures packets can be forwarded even when the buffer size is smaller than 1.
                self.token += self.rate * (time - self.depart)
                # Packet departure consumes a token.
                self.token -= 1
                self.depart = time
                if len(self.backlog) == 0:
                    # Terminate a busy period.
                    self.idle = True
                    return time, 0, 0, self.idle, forwarded_idx, forwarded_number, next_component
            # Examine the next packet.
            next_arrival, next_number = self.backlog[0]
            delay = 0
            if self.token < 1:
                delay = (1 - self.token) / self.rate
            next_depart = self.depart + delay
            return next_depart, 0, next_number, self.idle, forwarded_idx, forwarded_number, next_component
        else:
            # Release packets regardless of the token bucket state, only happens when the token bucket is not active.
            # Check if the packet to forward has the right sequence number.
            if self.active or len(self.backlog) - self.head_pointer == 0 or self.backlog[self.head_pointer][
                1] != packet_number:
                # Redundant forward event. Ignore.
                return time, 0, 0, True, 0, 0, None
            _, forwarded_number = self.backlog[self.head_pointer]
            forwarded_idx, next_component = self.component_idx, self.next
            # Update the head packet pointer.
            self.head_pointer += 1
            if self.head_pointer == len(self.backlog):
                # All backlogged packets have been released, stop forwarding packets.
                return time, 0, 0, True, forwarded_idx, forwarded_number, next_component
            # Forward the next backlogged packet.
            _, next_number = self.backlog[self.head_pointer]
            return time, 0, next_number, False, forwarded_idx, forwarded_number, next_component

    def peek(self, time):
        # Update the token bucket state.
        token = min(self.token + self.rate * (time - self.depart), self.buffer_size)
        return token, len(self.backlog) - self.head_pointer

    def activate(self, action):
        self.active = action
        return

    def reset(self):
        self.buffer_size = self.burst
        self.active = True
        self.backlog = []
        self.max_backlog_size = 0
        self.head_pointer = 0
        self.token = self.burst
        self.depart = 0
        super().reset()
        return


class ExtraTokenBucket(TokenBucket):

    def __init__(self, rate, burst, component_idx=0, internal=False):
        super().__init__(rate, burst, component_idx=component_idx, internal=internal)
        self.extra_token = 0
        return

    def forward(self, time, packet_number, component_idx, is_conformant):
        next_depart, next_idx, next_number, idle, forwarded_idx, forwarded_number, next_component = super().forward(
            time, packet_number, component_idx, is_conformant)
        # Update the tracker if extra token is used to forward the packet.
        if self.token < self.extra_token:
            self.extra_token = self.token
            self.buffer_size = self.burst + self.extra_token
        return next_depart, next_idx, next_number, idle, forwarded_idx, forwarded_number, next_component

    def reset(self):
        super().reset()
        self.extra_token = 0
        return


class PassiveExtraTokenBucket(ExtraTokenBucket):

    def add_token(self, time, token_num):
        token, _ = self.peek(time)
        self.token = token
        self.depart = time
        self.token += token_num
        # Update the extra token tracker and the buffer size as well.
        self.extra_token += token_num
        self.buffer_size = self.burst + self.extra_token
        return

    def reset_token(self, time):
        token, _ = self.peek(time)
        self.token = token
        self.depart = time
        # Remove the unused extra token from the token bucket.
        self.token -= self.extra_token
        self.extra_token = 0
        self.buffer_size = self.burst
        # Ensure the token number does not exceed the buffer size.
        self.token = min(self.token, self.burst)
        return


class DeSyncExtraTokenBucket(PassiveExtraTokenBucket):

    def __init__(self, rate, burst, latency_target, latency_min, flow_token_rate, backlog_window_size=0,
                 num_uniform_samples=10, min_num_inter_arrival_collect=20, local_protection_on=True,
                 local_protection_time=10.0, component_idx=0, internal=False):
        super().__init__(rate, burst, component_idx=component_idx, internal=internal)
        self.latency_target = latency_target
        self.latency_min = latency_min
        self.flow_token_rate = flow_token_rate
        self.backlog_window_size = backlog_window_size
        self.num_uniform_samples = num_uniform_samples
        self.min_num_inter_arrival_collect = min_num_inter_arrival_collect
        self.max_num_inter_arrival = min_num_inter_arrival_collect + 5
        self.local_protection_on = local_protection_on
        self.local_protection_time = local_protection_time
        self.average_wait_time_multiplier = 10
        self.extra_token_prob = 0
        self.waiting = False
        self.event_number = 0
        self.backlog_times = []
        self.backlog_samples = {}
        self.burst_inter_arrival_records = np.array([])
        self.last_burst_arrival = None
        self.local_protection_until = 0
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        backlog_old = len(self.backlog)
        super().arrive(time, packet_number, component_idx, is_internal)
        # Compute burst inter-arrival times.
        if self.last_burst_arrival is None:
            self.last_burst_arrival = time
            self.local_protection_until = time + self.local_protection_time
        else:
            # If inter-arrival time is smaller than the token generation time, considered as the same burst.
            inter_arrival = time - self.last_burst_arrival
            if inter_arrival >= 1 / self.flow_token_rate:
                self.last_burst_arrival = time
                normal_record = self.check_inter_arrival_record(inter_arrival)
                if not normal_record:
                    self.local_protection_until = time + self.local_protection_time
                self.add_inter_arrival_record(inter_arrival)
        # Keep track of status change in shaper backlog.
        if backlog_old != len(self.backlog):
            if self.last_burst_arrival not in self.backlog_samples.keys():
                self.backlog_times.append(self.last_burst_arrival)
                self.backlog_samples[self.last_burst_arrival] = len(self.backlog)
            self.backlog_samples[self.last_burst_arrival] = max(self.backlog_samples[self.last_burst_arrival],
                                                                len(self.backlog))
        return self.idle

    def forward(self, time, packet_number, component_idx, is_conformant):
        backlog_old = len(self.backlog)
        next_depart, next_idx, next_number, idle, forwarded_idx, forwarded_number, next_component = super().forward(
            time, packet_number, component_idx, is_conformant)
        # Reject scheduled extra tokens if the backlog is cleared before the tokens are granted.
        if len(self.backlog) == 0 and self.waiting:
            self.waiting = False
            self.event_number += 1
        # # Keep track of status change in shaper backlog.
        # if backlog_old != len(self.backlog):
        #     if time not in self.backlog_samples.keys():
        #         self.backlog_times.append(time)
        #     self.backlog_samples[time] = len(self.backlog)
        return next_depart, next_idx, next_number, idle, forwarded_idx, forwarded_number, next_component

    def schedule_extra_tokens(self, time):
        schedule_seed = np.random.rand()
        schedule_thresh = min(self.latency_min / self.latency_target * self.extra_token_prob, 1.0)
        if len(self.backlog) == 0 or self.waiting or (
                self.local_protection_on and time <= self.local_protection_until) or schedule_seed >= schedule_thresh:
            return 0, -1
        else:
            average_wait_time = self.latency_target * self.average_wait_time_multiplier
            assert average_wait_time >= 0
            # wait_time = 0 if average_wait_time == 0 else np.random.exponential(average_wait_time)
            wait_time = 0 if average_wait_time == 0 else np.random.uniform(low=0.0, high=2 * average_wait_time)
            self.waiting = True
            return self.event_number, wait_time

    def get_extra_tokens(self, time, event_number):
        if event_number != self.event_number:
            return False
        # Grant extra tokens to clear shaper backlog.
        self.reset_token(time)
        extra_token_num = len(self.backlog)
        self.add_token(time, extra_token_num)
        self.waiting = False
        self.event_number += 1
        return extra_token_num > 0

    def peek_backlog_samples(self, time):
        assert len(self.backlog_times) == 0 or self.backlog_times[-1] < time
        # Compress backlog records through uniform sampling.
        compressed_backlog = []
        start_idx = 0
        sample_window_size = self.backlog_window_size / self.num_uniform_samples
        for sample_idx, sample_time in enumerate(np.linspace(time - self.backlog_window_size, time,
                                                             num=self.num_uniform_samples, endpoint=False)):
            end_idx = bisect.bisect_left(self.backlog_times, sample_time + sample_window_size)
            # Keep the maximum backlog size observed within the sampling window.
            max_backlog = 0
            for time_idx in range(start_idx, end_idx):
                backlog_time = self.backlog_times[time_idx]
                backlog_value = self.backlog_samples[backlog_time]
                max_backlog = max(max_backlog, backlog_value)
            start_idx = end_idx
            # # Use the last backlog sample if no sample is available within the sampling window.
            # if max_backlog == -1:
            #     max_backlog = self.last_backlog
            # # Record a backlog update if state changes.
            # if max_backlog != self.last_compressed_backlog:
            #     self.last_compressed_backlog = max_backlog
            # Record non-zero backlog.
            if max_backlog > 0:
                compressed_backlog.append((sample_idx, max_backlog))
        # Remove recorded backlog samples.
        self.backlog_times = []
        self.backlog_samples = {}
        return compressed_backlog

    def check_inter_arrival_record(self, inter_arrival):
        # Remove inter-arrival records that are considered outliers using the 3-sigma rule.
        if len(self.burst_inter_arrival_records) >= self.min_num_inter_arrival_collect:
            while True:
                avg, std = np.mean(self.burst_inter_arrival_records), np.std(self.burst_inter_arrival_records)
                not_outlier = np.abs(avg - self.burst_inter_arrival_records) <= 3 * std
                self.burst_inter_arrival_records = self.burst_inter_arrival_records[not_outlier]
                if np.all(not_outlier):
                    break
        # Enforce local protection if not enough records have been collected.
        if len(self.burst_inter_arrival_records) < self.min_num_inter_arrival_collect:
            return False
        avg, std = np.mean(self.burst_inter_arrival_records), np.std(self.burst_inter_arrival_records)
        # Use the 3-sigma rule to check if the new inter-arrival time record comes from the same distribution.
        return np.abs(avg - inter_arrival) <= 3 * std

    def add_inter_arrival_record(self, inter_arrival):
        # Remove redundant inter-arrival records.
        num_delete = max(len(self.burst_inter_arrival_records) - self.max_num_inter_arrival + 1, 0)
        self.burst_inter_arrival_records = self.burst_inter_arrival_records[num_delete:]
        self.burst_inter_arrival_records = np.append(self.burst_inter_arrival_records, inter_arrival)
        return

    def reset(self):
        super().reset()
        self.average_wait_time_multiplier = 10
        self.extra_token_prob = 0
        self.waiting = False
        self.event_number = 0
        self.backlog_times = []
        self.backlog_samples = {}
        self.last_burst_arrival = None
        self.local_protection_until = 0
        return


class TokenBucketFluid:
    """Token Bucket Shaper that takes a fluid traffic arrival function as input and generates packetized output."""

    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self.token_count = [[], []]
        return

    def forward(self, arrival):
        # Sanity check on the traffic arrival function.
        assert all([nt <= ct for nt, ct in zip(arrival[0][:-1], arrival[0][1:])]), "Timestamps in the arrival " \
                                                                                   "functions should be non-decreasing."
        assert all([na <= ca for na, ca in zip(arrival[1][:-1], arrival[1][1:])]), "Traffic data in the arrival " \
                                                                                   "functions should be non-decreasing."
        assert len(arrival[0]) == len(arrival[1]), "Traffic arrival function should have the same number of " \
                                                   "timestamps and traffic data."
        assert len(arrival[0]) > 0, "Traffic arrival function should have at least 1 data point."
        assert arrival[0][0] == 0 and arrival[1][0] == 0, "Traffic arrival function should start from the origin."
        # Compute the token count as a function of time.
        time, data, token = 0, 0, self.burst
        for arrival_time, arrival_data in zip(arrival[0], arrival[1]):
            time_interval, data_interval = arrival_time - time, arrival_data - data
            token_increase = time_interval * self.rate
            expect_token = token - data_interval + token_increase
            if token_increase <= data_interval:
                assert expect_token >= -1e-5, "Traffic arrival function should conform with the token bucket profile."
            elif expect_token > self.burst:
                time_replenished = (self.burst - token) / (self.rate - data_interval / time_interval)
                time += time_replenished
                expect_token = self.burst
                self.token_count[0].append(time)
                self.token_count[1].append(self.burst)
            time, data, token = arrival_time, arrival_data, expect_token
            self.token_count[0].append(time)
            self.token_count[1].append(token)
        # Compute the departure time of packetized output.
        departure = []
        packet_count, arrival_idx = 1, 0
        while packet_count <= arrival[1][-1]:
            # Find the first left-most arrival_idx on which the traffic data >= packet_count.
            if packet_count > arrival[1][arrival_idx + 1]:
                arrival_idx += 1
                continue
            time_left, time_right = arrival[0][arrival_idx], arrival[0][arrival_idx + 1]
            traffic_left, traffic_right = arrival[1][arrival_idx], arrival[1][arrival_idx + 1]
            departure.append(
                time_left + (time_right - time_left) / (traffic_right - traffic_left) * (packet_count - traffic_left))
            packet_count += 1
        return departure

    def peek(self, time):
        assert time >= 0, "Should check token number at time >= 0."
        # Check the token number.
        peek_idx = bisect.bisect_left(self.token_count[0], time)
        if peek_idx < len(self.token_count[0]) and self.token_count[0][peek_idx] == time:
            return self.token_count[1][peek_idx]
        time_left = self.token_count[0][peek_idx - 1]
        token_left = self.token_count[1][peek_idx - 1]
        if peek_idx < len(self.token_count[0]):
            time_right = self.token_count[0][peek_idx]
            token_right = self.token_count[1][peek_idx]
            token = token_left + (token_right - token_left) / (time_right - time_left) * (time - time_left)
        else:
            token = min(token_left + self.rate * (time - time_left), self.burst)
        return token

    def reset(self):
        self.token_count = [[], []]
        return


class MultiSlopeShaper(NetworkComponent):

    def __init__(self, flow_idx, *args, ingress=False, internal=False):
        self.flow_idx = flow_idx
        # Set each token bucket from the input list.
        for tb_idx, tb in enumerate(args):
            assert isinstance(tb, PassiveExtraTokenBucket), "Every argument passed into MultiSlopeShaper " \
                                                            "must be a TokenBucket instance."
            tb.component_idx = tb_idx
            tb.internal = True
            tb.next = self
        self.ingress = ingress
        self.internal = internal
        self.token_buckets = args
        self.eligible_packets = [[] for _ in range(len(args))]
        super().__init__()
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        # A packet is eligible if released by all the token bucket shapers.
        self.eligible_packets[component_idx].append((time, packet_number))
        return all(len(ep) > 0 for ep in self.eligible_packets)

    def forward(self, time, packet_number, component_idx, is_conformant):
        # Release an eligible packet.
        forwarded_number = 0
        for ep in self.eligible_packets:
            _, forwarded_number = ep.pop(0)
        return time, 0, 0, True, self.flow_idx, forwarded_number, self.next

    def peek(self, time):
        # Return the maximum number of backlogged packets across all the token buckets.
        max_backlog = 0
        for tb in self.token_buckets:
            _, backlog = tb.peek(time)
            if backlog > max_backlog:
                max_backlog = backlog
        return max_backlog

    def activate(self, action):
        # Turn on or turn off all the token bucket shapers.
        for tb in self.token_buckets:
            tb.activate(action)
        return

    def add_token(self, time, tb_idx, token_num):
        # Add token to the specified token bucket shaper.
        self.token_buckets[tb_idx].add_token(time, token_num)
        return

    def reset_token(self, time, tb_idx):
        # Remove unused extra token.
        self.token_buckets[tb_idx].reset_token(time)
        return

    def reset(self):
        self.eligible_packets = [[] for _ in range(len(self.token_buckets))]
        for tb in self.token_buckets:
            tb.reset()
        super().reset()
        return


class InterleavedShaper(NetworkComponent):

    def __init__(self, packet_size, *args):
        self.packet_size = packet_size
        self.num_flow = len(packet_size)
        # Set each multi slope shaper from the input list.
        self.multi_slope_shapers = [None] * self.num_flow
        for ms_idx, ms in enumerate(args):
            assert isinstance(ms, MultiSlopeShaper), "Every argument passed into InterleavedShaper " \
                                                     "must be a MultiSlopeShaper instance."
            ms.internal = True
            self.multi_slope_shapers[ms.flow_idx] = ms
            ms.next = self
        self.backlog = []
        self.max_backlog_size = 0
        super().__init__()
        return

    def arrive(self, time, packet_count, component_idx, is_internal):
        if not is_internal:
            # Add the packet and its flow index to the backlog.
            self.backlog.append((component_idx, packet_count, False))
            self.max_backlog_size = max(self.max_backlog_size, self.peek(time))
            return False
        else:
            # Tag the specified non-eligible enqueued packet as eligible.
            packet_idx = self.backlog.index((component_idx, packet_count, False))
            self.backlog[packet_idx] = (component_idx, packet_count, True)
            # Forward the first packet if eligible.
            return packet_idx == 0

    def forward(self, time, packet_number, component_idx, is_conformant):
        # Check if the packet to forward has the right flow index and packet sequence number.
        if len(self.backlog) == 0 or (self.backlog[0][0] != component_idx or self.backlog[0][1] != packet_number):
            # Redundant forward event. Ignore.
            return time, 0, 0, True, 0, 0, None
        # Release the packet at the top of the queue.
        forwarded_idx, forwarded_number, eligible = self.backlog.pop(0)
        assert eligible, "Non-eligible packet forwarded."
        # Examine the next packet.
        next_idx, next_number, next_eligible = 0, 0, False
        if len(self.backlog) > 0:
            next_idx, next_number, next_eligible = self.backlog[0]
        return time, next_idx, next_number, not next_eligible, forwarded_idx, forwarded_number, self.next

    def peek(self, time):
        # Return the size of backlogged packets.
        backlog_flow = [packet[0] for packet in self.backlog]
        return np.sum(self.packet_size[backlog_flow])

    def reset(self):
        for ms in self.multi_slope_shapers:
            if ms is not None:
                ms.reset()
        self.backlog = []
        self.max_backlog_size = 0
        super().reset()
        return
