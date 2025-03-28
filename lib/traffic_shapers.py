from bisect import bisect_left
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


class ProactiveExtraTokenBucket(PassiveExtraTokenBucket):

    def __init__(self, rate, burst, original_tb, average_wait_time, packet_size, latency_target,
                 transmission_delay, propagation_delay, component_idx=0, internal=False):
        super().__init__(rate, burst, component_idx=component_idx, internal=internal)
        self.original_tb = original_tb
        assert average_wait_time > 0
        self.average_wait_time = average_wait_time
        self.packet_size = packet_size
        self.latency_target = latency_target
        assert isinstance(transmission_delay, np.ndarray)
        self.transmission_delay = transmission_delay
        self.num_link = len(transmission_delay)
        assert isinstance(propagation_delay, np.ndarray) and np.size(propagation_delay) == self.num_link
        self.propagation_delay = propagation_delay
        self.scheduler_backlog = np.zeros_like(transmission_delay)
        self.scheduler_utilization = np.zeros_like(transmission_delay)
        self.extra_waiting = False
        self.extra_eligible = False
        self.wait_time = 0
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        super().arrive(time, packet_number, component_idx, is_internal)
        if self.evaluate_feasibility(time) and not self.extra_waiting:
            self.extra_waiting = True
            self.extra_eligible = True
            self.wait_time = np.random.exponential(self.average_wait_time)
        else:
            self.extra_eligible = False
        return self.idle

    def evaluate_feasibility(self, time):
        if len(self.backlog) == 0:
            return False
        shaping_delay = np.array([time - pkt[0] for pkt in self.backlog])
        assert np.all(shaping_delay >= 0)
        # Check the remaining burst from the original token bucket profile.
        remaining_burst = int(self.original_tb.peek(time))
        packet_arrival_time = np.zeros((len(self.backlog) + remaining_burst,), dtype=float)
        for link_idx in range(len(self.transmission_delay)):
            # Compute the transmission time of each packet assuming the scheduler is Processor Sharing (PS) and the
            # link utilization remains unchanged. Set available bandwidth to be at least 1% to avoid infinite
            # transmission time.
            link_available_bandwidth = max(1 - self.scheduler_utilization[link_idx], 0.01)
            link_transmission_delay = self.transmission_delay[link_idx] / link_available_bandwidth
            # Compute the departure time of the previous packet assuming the scheduler backlog remains unchanged
            # upon the arrival of the first packet.
            prev_departure = packet_arrival_time[0] + link_transmission_delay * self.scheduler_backlog[link_idx]
            for packet_idx in range(len(packet_arrival_time)):
                # Compute the departure time of the packet.
                packet_arrival = packet_arrival_time[packet_idx]
                packet_departure = max(packet_arrival, prev_departure) + self.packet_size * link_transmission_delay
                prev_departure = packet_departure
                packet_departure += self.propagation_delay[link_idx]
                packet_arrival_time[packet_idx] = packet_departure
        packet_arrival_time[:len(shaping_delay)] += shaping_delay
        worst_end_to_end_delay = np.amax(packet_arrival_time)
        return worst_end_to_end_delay <= self.latency_target

    def update_state(self, scheduler_backlog, scheduler_utilization):
        assert isinstance(scheduler_backlog, np.ndarray) and np.size(scheduler_backlog) == self.num_link
        assert isinstance(scheduler_utilization, np.ndarray) and np.size(scheduler_utilization) == self.num_link
        assert np.all(0 <= scheduler_backlog)
        assert np.all(0 <= scheduler_utilization) and np.all(scheduler_utilization <= 1)
        self.scheduler_backlog = scheduler_backlog
        self.scheduler_utilization = scheduler_utilization
        return

    def clear_backlog(self, time):
        # Grant extra tokens to clear shaper backlog.
        self.reset_token(time)
        extra_token_num = len(self.backlog)
        self.add_token(time, extra_token_num)
        self.extra_waiting = False
        return

    def reset(self):
        super().reset()
        self.scheduler_backlog = np.zeros_like(self.transmission_delay)
        self.scheduler_utilization = np.zeros_like(self.transmission_delay)
        self.extra_waiting = False
        self.extra_eligible = False
        self.wait_time = 0
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
        peek_idx = bisect_left(self.token_count[0], time)
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
