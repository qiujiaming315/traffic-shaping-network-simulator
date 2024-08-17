from lib.traffic_shapers import NetworkComponent

class FIFOScheduler(NetworkComponent):

    def __init__(self, bandwidth, packet_size, buffer_size=None):
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.buffer_size = buffer_size
        self.num_flow = len(packet_size)
        self.backlog = []
        self.backlog_flow = []
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
            self.backlog.append((time, packet_number))
            self.backlog_flow.append(component_idx)
            self.max_backlog_size = max(self.max_backlog_size, self.peek(time))
        else:
            pass
        return self.idle

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
            _, packet_number = self.backlog.pop(0)
            flow_idx = self.backlog_flow.pop(0)
            next_component = self.next[flow_idx]
            if len(self.backlog) == 0:
                # Terminate a busy period.
                self.idle = True
                return time, self.idle, flow_idx, packet_number, next_component
        # Examine the next packet.
        next_arrival, _ = self.backlog[0]
        next_flow = self.backlog_flow[0]
        next_depart = max(next_arrival, self.depart) + self.packet_size[next_flow] / self.bandwidth
        return next_depart, self.idle, flow_idx, packet_number, next_component

    def peek(self, time):
        # Return the size of backlogged packets.
        return np.sum(self.packet_size[self.backlog_flow])

    def reset(self):
        self.backlog = []
        self.backlog_flow = []
        self.max_backlog_size = 0
        self.depart = 0
        super().reset()
        return