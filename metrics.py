def calculate_metrics(processes, total_time):
    avg_waiting = sum(p.waiting_time for p in processes) / len(processes)
    avg_turnaround = sum(p.turnaround_time for p in processes) / len(processes)

    cpu_busy = sum(p.burst_time for p in processes)
    cpu_utilization = (cpu_busy / total_time) * 100

    return avg_waiting, avg_turnaround, cpu_utilization
