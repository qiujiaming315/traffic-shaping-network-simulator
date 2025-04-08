import bisect
import heapq
import numpy as np
from dataclasses import dataclass

from lib.traffic_shapers import NetworkComponent


class Scheduler(NetworkComponent):
    """Parent class of network schedulers."""

    def __init__(self, bandwidth, packet_size, busy_period_window_size=0, max_backlog_window_size=0,
                 propagation_delay=0, buffer_size=None):
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.busy_period_window_size = busy_period_window_size
        self.max_backlog_window_size = max_backlog_window_size
        self.propagation_delay = propagation_delay
        self.buffer_size = buffer_size
        self.num_flow = len(packet_size)
        self.backlog = []
        self.max_backlog_size = 0
        self.max_backlog_size_recent = []
        self.busy_period = []
        self.depart = 0
        self.terminal = [False] * self.num_flow
        super().__init__()
        self.next = [None] * self.num_flow
        self.next_packet = None
        return

    def arrive(self, time, packet_number, component_idx, is_internal):
        # Check if the buffer has enough space to accommodate the packet.
        buffer_available = self.buffer_size is None or self.buffer_size >= self.packet_size[
            component_idx] + self.peek_backlog(time)
        # Add the packet and its flow index to the backlog.
        if buffer_available:
            self.add_packet(time, packet_number, component_idx)
            backlog_size = self.peek_backlog(time)
            self.max_backlog_size = max(self.max_backlog_size, backlog_size)
            # Update recent max backlog sliding window.
            self.update_recent_max_backlog(time)
            while len(self.max_backlog_size_recent) > 0:
                recent_backlog = self.max_backlog_size_recent[-1][1]
                if recent_backlog <= backlog_size:
                    self.max_backlog_size_recent.pop()
                else:
                    break
            self.max_backlog_size_recent.append((time, backlog_size))
        return self.idle

    def add_packet(self, time, packet_number, component_idx):
        # Function to add packets to the scheduler buffer.
        return

    def forward(self, time, packet_number, component_idx, is_conformant):
        # Check if the packet to forward has the right flow index and packet sequence number.
        if (len(self.backlog) == 0 and self.next_packet is None) or (self.next_packet is not None and (
                self.next_packet.flow_idx != component_idx or self.next_packet.packet_number != packet_number)):
            # Redundant forward event. Ignore.
            return time, 0, 0, self.idle, 0, 0, None
        # Update the last packet departure time.
        self.depart = time
        forwarded_number, forwarded_idx, next_component = 0, 0, None
        if self.idle:
            # Initiate a busy period.
            self.idle = False
            self.busy_period.append(time)
        else:
            # Release the forwarded packet.
            forwarded_number, forwarded_idx = self.next_packet.packet_number, self.next_packet.flow_idx
            next_component = self.next[forwarded_idx]
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                self.busy_period.append(time)
                self.next_packet = None
                return time, 0, 0, self.idle, forwarded_idx, forwarded_number, next_component
        # Examine the next packet.
        next_packet = heapq.heappop(self.backlog)
        next_arrival, next_idx, next_number = next_packet.arrival_time, next_packet.flow_idx, next_packet.packet_number
        self.next_packet = next_packet
        next_depart = max(next_arrival, self.depart) + self.packet_size[next_idx] / self.bandwidth
        return next_depart, next_idx, next_number, self.idle, forwarded_idx, forwarded_number, next_component

    def peek_backlog(self, time):
        # Return the size of backlogged packets.
        backlog_flow = [packet.flow_idx for packet in self.backlog]
        return np.sum(self.packet_size[backlog_flow])

    def peek_utlization(self, time):
        # Return the link utilization over the last busy period window.
        # Discard busy periods that end before the last window starts.
        window_idx = bisect.bisect(self.busy_period, time - self.busy_period_window_size)
        self.busy_period = self.busy_period[window_idx:]
        if window_idx % 2 == 1:
            # Reset the busy period start time to the start time of the last window.
            self.busy_period.insert(0, time - self.busy_period_window_size)
        busy_period_length = 0
        if len(self.busy_period) % 2 == 1:
            busy_period_length += time - self.busy_period[-1]
        for idx in range(len(self.busy_period) // 2):
            busy_period_length += self.busy_period[2 * idx + 1] - self.busy_period[2 * idx]
        # Compute the utlization.
        utilization = busy_period_length / self.busy_period_window_size
        return utilization

    def peek(self, time):
        return self.peek_backlog(time), self.peek_utlization(time)

    def update_recent_max_backlog(self, time):
        # Remove outdated backlog observations.
        while len(self.max_backlog_size_recent) > 0:
            backlog_time = self.max_backlog_size_recent[0][0]
            if backlog_time < time - self.max_backlog_window_size:
                self.max_backlog_size_recent.pop(0)
            else:
                break
        return

    def peek_recent_max_backlog(self, time):
        # Return the max backlog size over the last sliding window.
        self.update_recent_max_backlog(time)
        if len(self.max_backlog_size_recent) == 0:
            return self.peek_backlog(time)
        return self.max_backlog_size_recent[0][1]

    def reset(self):
        self.backlog = []
        self.max_backlog_size = 0
        self.max_backlog_size_recent = []
        self.busy_period = []
        self.depart = 0
        self.next_packet = None
        super().reset()
        return


class FIFOScheduler(Scheduler):
    """FIFO Scheduler."""

    def __init__(self, bandwidth, packet_size, busy_period_window_size=0, max_backlog_window_size=0,
                 propagation_delay=0, buffer_size=None):
        super().__init__(bandwidth, packet_size, busy_period_window_size=busy_period_window_size,
                         max_backlog_window_size=max_backlog_window_size, propagation_delay=propagation_delay,
                         buffer_size=buffer_size)
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

    def __init__(self, bandwidth, packet_size, offset, *args, busy_period_window_size=0, max_backlog_window_size=0,
                 propagation_delay=0, buffer_size=None):
        super().__init__(bandwidth, packet_size, busy_period_window_size=busy_period_window_size,
                         max_backlog_window_size=max_backlog_window_size, propagation_delay=propagation_delay,
                         buffer_size=buffer_size)
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

    def reset(self):
        super().reset()
        for ms_shaper in self.multi_slope_shapers:
            ms_shaper.reset()
        return


@dataclass
class Packet:
    rank: float
    arrival_time: float
    packet_number: int
    flow_idx: int

    def __lt__(self, other):
        return (self.rank, self.packet_number) < (other.rank, other.packet_number)
