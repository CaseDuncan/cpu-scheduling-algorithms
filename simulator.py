"""
Operating System Process Scheduler and Memory Manager Simulator
B206 Operating Systems - Individual Project
Author: [Your Name]
Student ID: [Your ID]
"""

import heapq
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time

# ==================== ENUM DEFINITIONS ====================

class ProcessState(Enum):
    """Represents possible states of a process"""
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"

class SchedulingAlgorithm(Enum):
    """Available scheduling algorithms"""
    FCFS = "FCFS"
    SJF = "SJF"
    ROUND_ROBIN = "ROUND_ROBIN"

class MemoryStrategy(Enum):
    """Available memory allocation strategies"""
    FIRST_FIT = "FIRST_FIT"
    BEST_FIT = "BEST_FIT"

# ==================== DATA STRUCTURES ====================

class MemoryBlock:
    """Represents a contiguous block of memory"""
    def __init__(self, start_address: int, size: int, is_free: bool = True, process_id: Optional[int] = None):
        self.start_address = start_address
        self.size = size
        self.is_free = is_free
        self.process_id = process_id
        self.next: Optional['MemoryBlock'] = None
    
    def __repr__(self):
        status = "Free" if self.is_free else f"Allocated(P{self.process_id})"
        return f"[Addr:{self.start_address:04d}, Size:{self.size:3d}MB, {status}]"

class Process:
    """Represents a process in the system"""
    def __init__(self, pid: int, arrival_time: int, burst_time: int, memory_req: int, priority: int = 0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.memory_req = memory_req
        self.priority = priority
        
        # State tracking
        self.state = ProcessState.NEW
        self.start_time: Optional[int] = None
        self.completion_time: Optional[int] = None
        self.waiting_time = 0
        self.memory_start: Optional[int] = None
        
        # For Gantt chart
        self.execution_intervals: List[Tuple[int, int]] = []
    
    def __lt__(self, other):
        """For priority queue comparisons"""
        return self.priority < other.priority
    
    def __repr__(self):
        return f"P{self.pid}(Arr:{self.arrival_time}, Burst:{self.burst_time}, Mem:{self.memory_req}MB)"

# ==================== MEMORY MANAGER ====================

class MemoryManager:
    """Manages memory allocation and deallocation"""
    def __init__(self, total_memory: int = 1024):  # 1GB default
        self.total_memory = total_memory
        self.used_memory = 0
        self.strategy = MemoryStrategy.FIRST_FIT
        
        # Initialize with one free block
        self.head = MemoryBlock(0, total_memory, True)
        self.memory_map: Dict[int, MemoryBlock] = {}  # PID -> MemoryBlock
    
    def set_strategy(self, strategy: MemoryStrategy):
        """Set memory allocation strategy"""
        self.strategy = strategy
    
    def allocate_memory(self, process: Process) -> bool:
        """Allocate memory for a process based on current strategy"""
        if self.strategy == MemoryStrategy.FIRST_FIT:
            return self._first_fit(process)
        else:
            return self._best_fit(process)
    
    def _first_fit(self, process: Process) -> bool:
        """First-fit memory allocation algorithm"""
        current = self.head
        
        while current:
            if current.is_free and current.size >= process.memory_req:
                return self._allocate_block(current, process)
            current = current.next
        
        # Try compaction if no suitable block found
        self._compact_memory()
        return self._first_fit(process)  # Try again after compaction
    
    def _best_fit(self, process: Process) -> bool:
        """Best-fit memory allocation algorithm"""
        best_block = None
        min_waste = float('inf')
        current = self.head
        
        while current:
            if current.is_free and current.size >= process.memory_req:
                waste = current.size - process.memory_req
                if waste < min_waste:
                    min_waste = waste
                    best_block = current
            current = current.next
        
        if best_block:
            return self._allocate_block(best_block, process)
        
        # Try compaction if no suitable block found
        self._compact_memory()
        return self._best_fit(process)
    
    def _allocate_block(self, block: MemoryBlock, process: Process) -> bool:
        """Allocate a specific block for a process"""
        # If block is larger than needed, split it
        if block.size > process.memory_req:
            remaining_size = block.size - process.memory_req
            new_block = MemoryBlock(
                block.start_address + process.memory_req,
                remaining_size,
                True
            )
            new_block.next = block.next
            block.next = new_block
            block.size = process.memory_req
        
        # Allocate the block
        block.is_free = False
        block.process_id = process.pid
        process.memory_start = block.start_address
        self.used_memory += process.memory_req
        self.memory_map[process.pid] = block
        
        return True
    
    def deallocate_memory(self, process_id: int):
        """Deallocate memory for a completed process"""
        if process_id not in self.memory_map:
            return
        
        block = self.memory_map[process_id]
        block.is_free = True
        block.process_id = None
        self.used_memory -= block.size
        
        # Remove from memory map
        del self.memory_map[process_id]
        
        # Merge with adjacent free blocks
        self._merge_free_blocks()
    
    def _merge_free_blocks(self):
        """Merge adjacent free memory blocks"""
        current = self.head
        
        while current and current.next:
            if current.is_free and current.next.is_free:
                # Merge current with next
                current.size += current.next.size
                current.next = current.next.next
            else:
                current = current.next
    
    def _compact_memory(self):
        """Compact memory by moving allocated blocks together"""
        print("  [Memory] Compacting memory...")
        allocated_blocks = []
        current = self.head
        
        # Collect all allocated blocks
        while current:
            if not current.is_free:
                allocated_blocks.append((current.start_address, current.size, current.process_id))
            current = current.next
        
        # Reconstruct memory from start
        current_address = 0
        self.head = None
        previous = None
        
        # Place allocated blocks first
        for start, size, pid in allocated_blocks:
            new_block = MemoryBlock(current_address, size, False, pid)
            
            if not self.head:
                self.head = new_block
            else:
                previous.next = new_block
            
            previous = new_block
            current_address += size
        
        # Add one big free block at the end
        if current_address < self.total_memory:
            free_block = MemoryBlock(current_address, self.total_memory - current_address, True)
            if previous:
                previous.next = free_block
            elif not self.head:
                self.head = free_block
    
    def get_fragmentation(self) -> float:
        """Calculate external fragmentation percentage"""
        free_blocks = 0
        total_free = 0
        current = self.head
        
        while current:
            if current.is_free:
                free_blocks += 1
                total_free += current.size
            current = current.next
        
        if total_free == 0:
            return 0.0
        
        # External fragmentation: many small free blocks that can't be used
        avg_free_size = total_free / free_blocks if free_blocks > 0 else 0
        max_process_size = max(block.size for block in self._get_free_blocks())
        
        if max_process_size < avg_free_size * 0.5:  # Small blocks relative to average
            return (free_blocks - 1) / free_blocks * 100 if free_blocks > 1 else 0
        return 0.0
    
    def _get_free_blocks(self):
        """Get all free memory blocks"""
        blocks = []
        current = self.head
        while current:
            if current.is_free:
                blocks.append(current)
            current = current.next
        return blocks
    
    def display_memory(self):
        """Display current memory layout"""
        print("\n" + "="*60)
        print("MEMORY LAYOUT")
        print("="*60)
        
        current = self.head
        while current:
            print(current)
            current = current.next
        
        utilization = (self.used_memory / self.total_memory) * 100
        print(f"\nMemory Utilization: {utilization:.1f}%")
        print(f"Fragmentation: {self.get_fragmentation():.1f}%")
        print("="*60)

# ==================== PROCESS SCHEDULER ====================

class ProcessScheduler:
    """Main scheduler that manages processes and CPU allocation"""
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.algorithm = SchedulingAlgorithm.FCFS
        self.time_quantum = 4  # For Round Robin
        
        # Process lists
        self.processes: List[Process] = []
        self.ready_queue: List[Process] = []
        self.completed_processes: List[Process] = []
        
        # System state
        self.current_time = 0
        self.running_process: Optional[Process] = None
        self.total_busy_time = 0
        self.gantt_chart: List[Tuple[int, int, int]] = []  # (start, end, pid)
        
        # Statistics
        self.stats = {
            'context_switches': 0,
            'cpu_idle_time': 0,
            'total_processes': 0
        }
    
    def set_algorithm(self, algorithm: SchedulingAlgorithm, time_quantum: int = 4):
        """Set scheduling algorithm and parameters"""
        self.algorithm = algorithm
        self.time_quantum = time_quantum
    
    def add_process(self, process: Process):
        """Add a new process to the system"""
        self.processes.append(process)
        self.stats['total_processes'] += 1
    
    def run_simulation(self):
        """Run the complete simulation"""
        print("\n" + "="*60)
        print(f"STARTING SIMULATION - Algorithm: {self.algorithm.value}")
        print("="*60)
        
        # Sort processes by arrival time
        self.processes.sort(key=lambda p: p.arrival_time)
        
        while self.processes or self.ready_queue or self.running_process:
            # Add arriving processes to ready queue
            self._add_arriving_processes()
            
            # Handle process completion
            if self.running_process and self.running_process.remaining_time == 0:
                self._complete_process()
            
            # Select next process to run
            if not self.running_process and self.ready_queue:
                self._schedule_next_process()
            
            # Execute current process
            self._execute_time_unit()
            
            # Update waiting times for processes in ready queue
            for process in self.ready_queue:
                process.waiting_time += 1
            
            # Move to next time unit
            self.current_time += 1
        
        print(f"\nSimulation completed at time {self.current_time}")
    
    def _add_arriving_processes(self):
        """Add processes that have arrived to ready queue"""
        arriving = [p for p in self.processes if p.arrival_time == self.current_time]
        
        for process in arriving:
            # Try to allocate memory
            if self.memory_manager.allocate_memory(process):
                process.state = ProcessState.READY
                self.ready_queue.append(process)
                self.processes.remove(process)
                print(f"Time {self.current_time:3d}: {process} arrived and allocated memory")
            else:
                print(f"Time {self.current_time:3d}: {process} arrived but NO MEMORY AVAILABLE")
                # Process stays in processes list, will retry later
    
    def _schedule_next_process(self):
        """Select next process based on scheduling algorithm"""
        if not self.ready_queue:
            return
        
        if self.algorithm == SchedulingAlgorithm.FCFS:
            self.running_process = self.ready_queue.pop(0)
        elif self.algorithm == SchedulingAlgorithm.SJF:
            self.ready_queue.sort(key=lambda p: (p.remaining_time, p.arrival_time))
            self.running_process = self.ready_queue.pop(0)
        elif self.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            self.running_process = self.ready_queue.pop(0)
        
        # Update process state
        self.running_process.state = ProcessState.RUNNING
        if self.running_process.start_time is None:
            self.running_process.start_time = self.current_time
        
        self.stats['context_switches'] += 1
        print(f"Time {self.current_time:3d}: Context switch to {self.running_process}")
    
    def _execute_time_unit(self):
        """Execute one time unit of the current process"""
        if self.running_process:
            # Process is running
            self.running_process.remaining_time -= 1
            self.total_busy_time += 1
            
            # Record execution for Gantt chart
            if not self.running_process.execution_intervals or \
               self.running_process.execution_intervals[-1][1] < self.current_time:
                self.running_process.execution_intervals.append((self.current_time, self.current_time + 1))
            else:
                # Extend last interval
                last_start, last_end = self.running_process.execution_intervals[-1]
                self.running_process.execution_intervals[-1] = (last_start, self.current_time + 1)
            
            # Check for preemption (Round Robin only)
            if self.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
                execution_duration = self.current_time - (self.running_process.execution_intervals[-1][0])
                if execution_duration >= self.time_quantum and self.running_process.remaining_time > 0:
                    # Preempt process
                    self.running_process.state = ProcessState.READY
                    self.ready_queue.append(self.running_process)
                    print(f"Time {self.current_time:3d}: {self.running_process} preempted (Time quantum expired)")
                    self.running_process = None
        else:
            # CPU is idle
            self.stats['cpu_idle_time'] += 1
            print(f"Time {self.current_time:3d}: CPU idle")
    
    def _complete_process(self):
        """Complete the currently running process"""
        if not self.running_process:
            return
        
        process = self.running_process
        process.completion_time = self.current_time
        process.state = ProcessState.TERMINATED
        
        # Free memory
        self.memory_manager.deallocate_memory(process.pid)
        
        # Add to completed list
        self.completed_processes.append(process)
        
        print(f"Time {self.current_time:3d}: {process} completed. "
              f"Turnaround: {process.completion_time - process.arrival_time}")
        
        # Record in Gantt chart
        self.gantt_chart.append((process.start_time or 0, 
                                process.completion_time, 
                                process.pid))
        
        self.running_process = None
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.completed_processes:
            return {}
        
        total_waiting = sum(p.waiting_time for p in self.completed_processes)
        total_turnaround = sum(p.completion_time - p.arrival_time 
                              for p in self.completed_processes)
        total_response = sum((p.start_time or 0) - p.arrival_time 
                           for p in self.completed_processes)
        
        avg_waiting = total_waiting / len(self.completed_processes)
        avg_turnaround = total_turnaround / len(self.completed_processes)
        avg_response = total_response / len(self.completed_processes)
        
        cpu_utilization = (self.total_busy_time / self.current_time) * 100
        throughput = len(self.completed_processes) / self.current_time
        
        return {
            'avg_waiting_time': avg_waiting,
            'avg_turnaround_time': avg_turnaround,
            'avg_response_time': avg_response,
            'cpu_utilization': cpu_utilization,
            'throughput': throughput,
            'context_switches': self.stats['context_switches'],
            'cpu_idle_time': self.stats['cpu_idle_time']
        }
    
    def display_gantt_chart(self):
        """Display ASCII Gantt chart"""
        if not self.gantt_chart:
            print("\nNo processes completed yet.")
            return
        
        print("\n" + "="*60)
        print("GANTT CHART")
        print("="*60)
        
        # Find time range
        min_time = min(start for start, _, _ in self.gantt_chart)
        max_time = max(end for _, end, _ in self.gantt_chart)
        
        # Print time line
        print("Time: ", end="")
        for t in range(min_time, max_time + 1):
            print(f"{t:3d}", end="")
        print()
        
        # Print process bars
        for start, end, pid in sorted(self.gantt_chart, key=lambda x: x[0]):
            print(f"P{pid:2d}: ", end="")
            for t in range(min_time, max_time + 1):
                if start <= t < end:
                    print("███", end="")
                else:
                    print("   ", end="")
            print(f"  ({start}-{end})")
        
        print("="*60)
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        
        # Display process details
        print("\nProcess Details:")
        print("-"*60)
        print("PID | Arrival | Burst | Start | Finish | Waiting | Turnaround")
        print("-"*60)
        
        for process in sorted(self.completed_processes, key=lambda p: p.pid):
            turnaround = process.completion_time - process.arrival_time
            waiting = turnaround - process.burst_time
            
            print(f"P{process.pid:2d} | "
                  f"{process.arrival_time:7d} | "
                  f"{process.burst_time:5d} | "
                  f"{process.start_time or 0:5d} | "
                  f"{process.completion_time or 0:6d} | "
                  f"{waiting:7d} | "
                  f"{turnaround:10d}")
        
        # Display metrics
        metrics = self.calculate_metrics()
        print("\n" + "-"*60)
        print("Performance Metrics:")
        print("-"*60)
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if metric in ['cpu_utilization', 'throughput']:
                    print(f"{metric.replace('_', ' ').title():25s}: {value:.2f}{'%' if metric == 'cpu_utilization' else ''}")
                else:
                    print(f"{metric.replace('_', ' ').title():25s}: {value:.2f}")
            else:
                print(f"{metric.replace('_', ' ').title():25s}: {value}")

# ==================== COMMAND LINE INTERFACE ====================

class OS_Simulator_CLI:
    """Command Line Interface for the OS Simulator"""
    def __init__(self):
        self.memory_manager = MemoryManager(total_memory=512)  # 512MB for simulation
        self.scheduler = ProcessScheduler(self.memory_manager)
        self.running = True
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("OPERATING SYSTEM SIMULATOR")
        print("="*60)
        print(f"Current Settings:")
        print(f"  Algorithm: {self.scheduler.algorithm.value}")
        print(f"  Memory Strategy: {self.memory_manager.strategy.value}")
        print(f"  Time Quantum: {self.scheduler.time_quantum}")
        print("-"*60)
        print("1. Add Process")
        print("2. Load Sample Processes")
        print("3. Run Simulation")
        print("4. Change Scheduling Algorithm")
        print("5. Change Memory Allocation Strategy")
        print("6. Display Memory Layout")
        print("7. Display Results")
        print("8. Display Gantt Chart")
        print("9. Reset Simulation")
        print("0. Exit")
        print("="*60)
    
    def add_process_ui(self):
        """UI for adding a process"""
        try:
            pid = len(self.scheduler.processes) + len(self.scheduler.completed_processes) + 1
            print(f"\nAdding Process P{pid}")
            
            arrival = int(input("Arrival Time: ") or "0")
            burst = int(input("Burst Time: ") or "5")
            memory = int(input("Memory Required (MB): ") or "64")
            priority = int(input("Priority (0=high, 5=low): ") or "0")
            
            process = Process(pid, arrival, burst, memory, priority)
            self.scheduler.add_process(process)
            
            print(f"Process P{pid} added successfully!")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    def load_sample_processes(self):
        """Load a set of sample processes for testing"""
        sample_processes = [
            Process(1, 0, 5, 100, 0),
            Process(2, 1, 3, 50, 1),
            Process(3, 2, 8, 200, 2),
            Process(4, 3, 2, 80, 0),
            Process(5, 4, 4, 120, 1),
            Process(6, 5, 6, 70, 2),
            Process(7, 6, 3, 90, 0),
            Process(8, 7, 5, 150, 1),
        ]
        
        for process in sample_processes:
            self.scheduler.add_process(process)
        
        print(f"\nLoaded {len(sample_processes)} sample processes.")
    
    def change_algorithm(self):
        """Change scheduling algorithm"""
        print("\nSelect Scheduling Algorithm:")
        print("1. FCFS (First-Come-First-Served)")
        print("2. SJF (Shortest Job First)")
        print("3. Round Robin")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == "1":
            self.scheduler.set_algorithm(SchedulingAlgorithm.FCFS)
            print("Algorithm set to FCFS")
        elif choice == "2":
            self.scheduler.set_algorithm(SchedulingAlgorithm.SJF)
            print("Algorithm set to SJF")
        elif choice == "3":
            time_quantum = input("Enter time quantum (default=4): ")
            quantum = int(time_quantum) if time_quantum else 4
            self.scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, quantum)
            print(f"Algorithm set to Round Robin with time quantum {quantum}")
        else:
            print("Invalid choice. Using FCFS.")
    
    def change_memory_strategy(self):
        """Change memory allocation strategy"""
        print("\nSelect Memory Allocation Strategy:")
        print("1. First Fit")
        print("2. Best Fit")
        
        choice = input("Enter choice (1-2): ")
        
        if choice == "1":
            self.memory_manager.set_strategy(MemoryStrategy.FIRST_FIT)
            print("Memory strategy set to First Fit")
        elif choice == "2":
            self.memory_manager.set_strategy(MemoryStrategy.BEST_FIT)
            print("Memory strategy set to Best Fit")
        else:
            print("Invalid choice. Using First Fit.")
    
    def run(self):
        """Main CLI loop"""
        print("Operating System Simulator Initialized!")
        print("Total Memory: 512MB")
        
        while self.running:
            self.display_menu()
            choice = input("\nEnter your choice (0-9): ")
            
            if choice == "1":
                self.add_process_ui()
            elif choice == "2":
                self.load_sample_processes()
            elif choice == "3":
                if not self.scheduler.processes and not self.scheduler.completed_processes:
                    print("No processes to simulate. Add processes first.")
                else:
                    self.scheduler.run_simulation()
            elif choice == "4":
                self.change_algorithm()
            elif choice == "5":
                self.change_memory_strategy()
            elif choice == "6":
                self.memory_manager.display_memory()
            elif choice == "7":
                self.scheduler.display_results()
            elif choice == "8":
                self.scheduler.display_gantt_chart()
            elif choice == "9":
                self.reset_simulation()
            elif choice == "0":
                print("\nThank you for using the OS Simulator!")
                self.running = False
            else:
                print("Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.memory_manager = MemoryManager(total_memory=512)
        self.scheduler = ProcessScheduler(self.memory_manager)
        print("\nSimulation reset to initial state.")

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run the simulator"""
    print("="*70)
    print("OS PROCESS SCHEDULER & MEMORY MANAGER SIMULATOR")
    print("B206 Operating Systems - Individual Project")
    print("="*70)
    
    # Create and run the CLI
    simulator = OS_Simulator_CLI()
    simulator.run()

if __name__ == "__main__":
    main()