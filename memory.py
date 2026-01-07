class MemoryBlock:
    def __init__(self, start, size):
        self.start = start
        self.size = size
        self.free = True
        self.pid = None


class MemoryManager:
    def __init__(self, total_memory):
        self.blocks = [MemoryBlock(0, total_memory)]
        self.total_memory = total_memory

    def allocate_first_fit(self, process):
        for block in self.blocks:
            if block.free and block.size >= process.memory_required:
                self._split_block(block, process)
                return True
        return False

    def allocate_best_fit(self, process):
        candidates = [
            block for block in self.blocks
            if block.free and block.size >= process.memory_required
        ]
        if not candidates:
            return False

        best_block = min(candidates, key=lambda b: b.size)
        self._split_block(best_block, process)
        return True

    def _split_block(self, block, process):
        allocated = MemoryBlock(block.start, process.memory_required)
        allocated.free = False
        allocated.pid = process.pid

        remaining = block.size - process.memory_required
        self.blocks.remove(block)

        self.blocks.append(allocated)
        if remaining > 0:
            self.blocks.append(
                MemoryBlock(block.start + process.memory_required, remaining)
            )

    def deallocate(self, pid):
        for block in self.blocks:
            if not block.free and block.pid == pid:
                block.free = True
                block.pid = None
        self._merge_free_blocks()

    def _merge_free_blocks(self):
        self.blocks.sort(key=lambda b: b.start)
        merged = []
        for block in self.blocks:
            if merged and merged[-1].free and block.free:
                merged[-1].size += block.size
            else:
                merged.append(block)
        self.blocks = merged

    def utilization(self):
        used = sum(block.size for block in self.blocks if not block.free)
        return (used / self.total_memory) * 100
