from process import Process
from scheduler import Scheduler
from memory import MemoryManager
from metrics import calculate_metrics


def input_processes():
    processes = []
    n = int(input("\nEnter number of processes: "))

    for i in range(n):
        print(f"\nProcess {i + 1}")
        arrival = int(input("Arrival time: "))
        burst = int(input("Burst time: "))
        memory = int(input("Memory required: "))

        processes.append(Process(i + 1, arrival, burst, memory))

    return processes


def select_memory_strategy(memory_manager, processes):
    print("\nSelect Memory Allocation Strategy:")
    print("1. First-Fit")
    print("2. Best-Fit")

    choice = input("Choice: ")

    for p in processes:
        if choice == "1":
            success = memory_manager.allocate_first_fit(p)
        elif choice == "2":
            success = memory_manager.allocate_best_fit(p)
        else:
            print("Invalid memory strategy.")
            return False

        if not success:
            print(f"Process {p.pid} rejected due to insufficient memory.")

    return True


def select_scheduling_strategy(processes):
    scheduler = Scheduler(processes)

    print("\nSelect CPU Scheduling Algorithm:")
    print("1. First Come First Served (FCFS)")
    print("2. Shortest Job First (SJF)")
    print("3. Round Robin")
    print("4. Exit Program")

    choice = input("Choice: ")

    if choice == "1":
        scheduler.fcfs()
    elif choice == "2":
        scheduler.sjf()
    elif choice == "3":
        quantum = int(input("Enter time quantum: "))
        scheduler.round_robin(quantum)
    elif choice == "4":
        return None
    else:
        print("Invalid choice.")
        return None

    return scheduler


def display_results(scheduler, memory_manager):
    avg_wt, avg_tat, cpu_util = calculate_metrics(
        scheduler.completed, scheduler.time
    )

    print("\nExecution Results:")
    print("-" * 40)
    for p in scheduler.completed:
        print(
            f"P{p.pid} | Waiting Time: {p.waiting_time} | "
            f"Turnaround Time: {p.turnaround_time}"
        )

    print("-" * 40)
    print(f"Average Waiting Time: {avg_wt:.2f}")
    print(f"Average Turnaround Time: {avg_tat:.2f}")
    print(f"CPU Utilization: {cpu_util:.2f}%")
    print(f"Memory Utilization: {memory_manager.utilization():.2f}%")


def main():
    print("=== Operating System Scheduler & Memory Simulator ===")

    while True:
        total_memory = int(input("\nEnter total system memory: "))
        memory_manager = MemoryManager(total_memory)

        processes = input_processes()

        if not select_memory_strategy(memory_manager, processes):
            continue

        scheduler = select_scheduling_strategy(processes)

        if scheduler is None:
            print("\nExiting program.")
            break

        display_results(scheduler, memory_manager)

        again = input("\nRun another simulation? (y/n): ").lower()
        if again != "y":
            print("\nExiting program.")
            break


if __name__ == "__main__":
    main()
