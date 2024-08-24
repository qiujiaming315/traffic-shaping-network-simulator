import heapq
import numpy as np
from dataclasses import dataclass

from lib.traffic_shapers import NetworkComponent


class Scheduler(NetworkComponent):
    """Parent class of network schedulers."""

    def __init__(self, bandwidth, packet_size, buffer_size=None):
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.buffer_size = buffer_size
        self.num_flow = len(packet_size)
        self.backlog = []
        self.max_backlog_size = 0
        self.depart = 0
        self.terminal = [False] * self.num_flow
        super().__init__()
        self.next = [None] * self.num_flow
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        # Check if the buffer has enough space to accommodate the packet.
        buffer_available = self.buffer_size is None or self.buffer_size >= self.packet_size[component_idx] + self.peek(
            time)
        # Add the packet and its flow index to the backlog.
        if buffer_available:
            self.add_packet(time, packet_number, component_idx)
            self.max_backlog_size = max(self.max_backlog_size, self.peek(time))
        return self.idle

    def add_packet(self, time, packet_number, component_idx):
        # Function to add packets to the scheduler buffer.
        return

    def forward(self, time):
        if len(self.backlog) == 0:
            # Redundant forward event. Ignore.
            return time, self.idle, 0, 0, None
        # Update the last packet departure time.
        self.depart = time
        packet_number, flow_idx, next_component = 0, 0, None
        if self.idle:
            # Initiate a busy period.
            self.idle = False
        else:
            # Release the forwarded packet.
            packet = heapq.heappop(self.backlog)
            packet_number, flow_idx = packet.packet_number, packet.flow_idx
            next_component = self.next[flow_idx]
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                return time, self.idle, flow_idx, packet_number, next_component
        # Examine the next packet.
        packet = self.backlog[0]
        next_arrival, next_flow = packet.arrival_time, packet.flow_idx
        next_depart = max(next_arrival, self.depart) + self.packet_size[next_flow] / self.bandwidth
        return next_depart, self.idle, flow_idx, packet_number, next_component

    def peek(self, time):
        # Return the size of backlogged packets.
        backlog_flow = [packet.flow_idx for packet in self.backlog]
        return np.sum(self.packet_size[backlog_flow])

    def reset(self):
        self.backlog = []
        self.max_backlog_size = 0
        self.depart = 0
        super().reset()
        return


class FIFOScheduler(Scheduler):
    """FIFO Scheduler."""

    def __init__(self, bandwidth, packet_size, buffer_size=None):
        super().__init__(bandwidth, packet_size, buffer_size=buffer_size)
        return

    def add_packet(self, time, packet_number, component_idx):
        # Sort packet according to their arrival time.
        new_packet = Packet(time, time, packet_number, component_idx)
        heapq.heappush(self.backlog, new_packet)
        return


class TokenBucketSCED:
    """Internal Token Bucket Shaper used by SCED to compute packet eligibility time."""

    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self.token = burst
        self.depart = 0
        super().__init__()
        return

    def get_eligibility_time(self, time):
        # Compute the eligibility time of a packet arriving at a given time.
        if time > self.depart:
            # Update the token number.
            self.token += self.rate * (time - self.depart)
            self.token = min(self.burst, self.token)
        eligibility_time = max(time, self.depart)
        if self.token < 1:
            eligibility_time += (1 - self.token) / self.rate
            self.token = 1
        self.depart = eligibility_time
        self.token -= 1
        return eligibility_time

    def reset(self):
        self.token = self.burst
        self.depart = 0
        return


class MultiSlopeShaperSCED:
    """Internal Multi-Slope Shaper used by SCED to compute packet eligibility time."""

    def __init__(self, *args):
        # Set each token bucket from the input list.
        for tb in args:
            assert isinstance(tb, TokenBucketSCED), "Every argument passed into MultiSlopeShaperSCED " \
                                                    "must be a TokenBucketSCED instance."
        self.token_buckets = args
        super().__init__()
        return

    def get_eligibility_time(self, time):
        # Compute the eligibility time of a packet arriving at a given time.
        eligibility_time = time
        for tb in self.token_buckets:
            eligibility_time = max(eligibility_time, tb.get_eligibility_time(time))
        return eligibility_time

    def reset(self):
        for tb in self.token_buckets:
            tb.reset()
        return


class SCEDScheduler(Scheduler):
    """SCED scheduler."""

    def __init__(self, bandwidth, packet_size, offset, *args, buffer_size=None):
        super().__init__(bandwidth, packet_size, buffer_size=buffer_size)
        self.multi_slope_shapers = [MultiSlopeShaperSCED()] * self.num_flow
        for flow_idx, ms in args:
            assert isinstance(ms, MultiSlopeShaperSCED), "Every argument passed into SCEDScheduler " \
                                                         "must be a MultiSlopeShaperSCED instance."
            self.multi_slope_shapers[flow_idx] = ms
        self.offset = offset
        return

    def add_packet(self, time, packet_number, component_idx):
        # Sort packet according to their eligibility time.
        packet_eligibility_time = self.multi_slope_shapers[component_idx].get_eligibility_time(time)
        packet_eligibility_time += self.offset[component_idx]
        new_packet = Packet(packet_eligibility_time, time, packet_number, component_idx)
        heapq.heappush(self.backlog, new_packet)
        return


@dataclass
class Packet:
    rank: float
    arrival_time: float
    packet_number: int
    flow_idx: int

    def __lt__(self, other):
        return self.rank < other.rank
