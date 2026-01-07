class Process:
    def __init__(self, pid, arrival_time, burst_time, memory_required):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.memory_required = memory_required

        self.start_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.turnaround_time = 0

    def __repr__(self):
        return f"P{self.pid}"
