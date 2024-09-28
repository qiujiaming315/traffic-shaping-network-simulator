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

    def forward(self, time):
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
        self.active = True
        self.backlog = []
        self.max_backlog_size = 0
        self.token = burst
        self.depart = 0
        super().__init__()
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        self.backlog.append((time, packet_number))
        self.max_backlog_size = max(self.max_backlog_size, len(self.backlog))
        return self.idle

    def forward(self, time):
        if len(self.backlog) == 0:
            # Redundant forward event. Ignore.
            return time, self.idle, 0, 0, None
        packet_number, component_idx, next_component = 0, 0, None
        if self.idle:
            # Initiate a busy period.
            if self.active:
                token, _ = self.peek(time)
                self.token = token
                self.depart = time
            self.idle = False
        else:
            # Release the forwarded packet.
            _, packet_number = self.backlog.pop(0)
            if self.active:
                self.token += self.rate * (time - self.depart) - 1
                self.depart = time
            component_idx, next_component = self.component_idx, self.next
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                return time, self.idle, component_idx, packet_number, next_component
        # Examine the next packet.
        next_arrival, _ = self.backlog[0]
        next_depart = time
        if self.active:
            delay = 0
            if self.token < 1:
                delay = (1 - self.token) / self.rate
            next_depart = max(next_arrival, self.depart) + delay
        return next_depart, self.idle, component_idx, packet_number, next_component

    def peek(self, time):
        # Update the token bucket state.
        token = min(self.token + self.rate * (time - self.depart), self.burst) if self.idle else 0
        return token, len(self.backlog)

    def activate(self, action, time):
        if action != self.active:
            if self.active:
                token, _ = self.peek(time)
                self.token = token
            self.depart = time
        self.active = action
        return

    def reset(self):
        self.active = True
        self.backlog = []
        self.max_backlog_size = 0
        self.token = self.burst
        self.depart = 0
        super().reset()
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

    def __init__(self, flow_idx, *args, internal=False):
        self.flow_idx = flow_idx
        # Set each token bucket from the input list.
        for tb_idx, tb in enumerate(args):
            assert isinstance(tb, TokenBucket), "Every argument passed into MultiSlopeShaper " \
                                                "must be a TokenBucket instance."
            tb.component_idx = tb_idx
            tb.internal = True
            tb.next = self
        self.internal = internal
        self.token_buckets = args
        self.eligible_packets = [[] for _ in range(len(args))]
        super().__init__()
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        # A packet is eligible if released by all the token bucket shapers.
        self.eligible_packets[component_idx].append((time, packet_number))
        return all(len(ep) > 0 for ep in self.eligible_packets)

    def forward(self, time):
        # Release an eligible packet.
        packet_number = 0
        for ep in self.eligible_packets:
            _, packet_number = ep.pop(0)
        return time, True, self.flow_idx, packet_number, self.next

    def peek(self, time):
        # Return the maximum number of backlogged packets across all the token buckets.
        max_backlog = 0
        for tb in self.token_buckets:
            _, backlog = tb.peek(time)
            if backlog > max_backlog:
                max_backlog = backlog
        return max_backlog

    def activate(self, action, time):
        # Turn on or turn off all the token bucket shapers.
        for tb in self.token_buckets:
            tb.activate(action, time)
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

    def forward(self, time):
        if len(self.backlog) == 0:
            # Redundant forward event. Ignore.
            return time, True, 0, 0, None
        # Release the packet at the top of the queue.
        flow_idx, packet_number, eligible = self.backlog.pop(0)
        assert eligible, "Non-eligible packet forwarded."
        # Examine the next packet.
        next_eligible = len(self.backlog) > 0 and self.backlog[0][2]
        return time, not next_eligible, flow_idx, packet_number, self.next

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
