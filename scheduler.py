from collections import deque


class Scheduler:
    def __init__(self, processes):
        self.processes = sorted(processes, key=lambda p: p.arrival_time)
        self.time = 0
        self.completed = []

    def fcfs(self):
        for process in self.processes:
            self.time = max(self.time, process.arrival_time)
            process.start_time = self.time
            process.finish_time = self.time + process.burst_time
            self.time += process.burst_time
            self._finalize(process)

    def sjf(self):
        ready = []
        processes = self.processes.copy()

        while processes or ready:
            ready.extend([p for p in processes if p.arrival_time <= self.time])
            processes = [p for p in processes if p.arrival_time > self.time]

            if not ready:
                self.time += 1
                continue

            ready.sort(key=lambda p: p.burst_time)
            process = ready.pop(0)

            process.start_time = self.time
            self.time += process.burst_time
            process.finish_time = self.time
            self._finalize(process)

    def round_robin(self, quantum):
        queue = deque()
        processes = self.processes.copy()

        while processes or queue:
            for p in processes[:]:
                if p.arrival_time <= self.time:
                    queue.append(p)
                    processes.remove(p)

            if not queue:
                self.time += 1
                continue

            process = queue.popleft()

            if process.start_time is None:
                process.start_time = self.time

            execution = min(quantum, process.remaining_time)
            process.remaining_time -= execution
            self.time += execution

            if process.remaining_time == 0:
                process.finish_time = self.time
                self._finalize(process)
            else:
                queue.append(process)

    def _finalize(self, process):
        process.turnaround_time = process.finish_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.burst_time
        self.completed.append(process)
