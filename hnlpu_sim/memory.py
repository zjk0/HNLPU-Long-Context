import numpy as np

class Memory:
    def __init__(self, size_byte, bandwidth_byte_per_s, fixed_access_latency_s):
        self._validate_integer(size_byte, "size_byte", minimum = 1)
        self._validate_number(bandwidth_byte_per_s, "bandwidth_byte_per_s", minimum = 0, inclusive = False)
        self._validate_number(fixed_access_latency_s, "fixed_access_latency_s", minimum = 0)

        self.size_byte = size_byte
        self.bandwidth_byte_per_s = bandwidth_byte_per_s
        self.fixed_access_latency_s = fixed_access_latency_s
        self.usage_byte = 0
        self.allocate_info = {}

    @staticmethod
    def _validate_integer(value, name, minimum = 0):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        if value < minimum:
            raise ValueError(f"{name} must be greater than or equal to {minimum}.")

    @staticmethod
    def _validate_number(value, name, minimum = 0, inclusive = True):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number.")
        if (inclusive and value < minimum) or (not inclusive and value <= minimum):
            comparison = "greater than or equal to" if inclusive else "greater than"
            raise ValueError(f"{name} must be {comparison} {minimum}.")

    @staticmethod
    def _validate_allocation_id(allocation_id, name):
        if allocation_id is None or (
            isinstance(allocation_id, str) and not allocation_id.strip()
        ):
            raise ValueError(f"{name} must not be empty.")
        try:
            hash(allocation_id)
        except TypeError as exc:
            raise TypeError(f"{name} must be hashable.") from exc
    
    def get_usage(self):
        return self.usage_byte
    
    def get_remaining_size(self):
        return self.size_byte - self.usage_byte

    def get_usage_ratio(self):
        return self.usage_byte / self.size_byte
        
    def is_full(self):
        if self.usage_byte >= self.size_byte:
            return True
        else:
            return False
        
    def is_empty(self):
        if self.usage_byte == 0:
            return True
        else:
            return False
        
    def allocate_memory(self, allocate_size_byte, allocate_id):
        self.ensure_consistent()
        self._validate_integer(allocate_size_byte, "allocate_size_byte", minimum = 1)
        self._validate_allocation_id(allocate_id, "allocate_id")
        
        # Allocate ID can not repeat.
        if allocate_id in self.allocate_info:
            raise ValueError(f"allocate_id({allocate_id}) has already existed.")
        
        if self.usage_byte + allocate_size_byte > self.size_byte:
            return False
        
        self.usage_byte += allocate_size_byte
        self.allocate_info[allocate_id] = allocate_size_byte
        self.ensure_consistent()
        return True
        
    def free_memory(self, free_id):
        self.ensure_consistent()
        self._validate_allocation_id(free_id, "free_id")

        # The ID to free must exist
        if free_id not in self.allocate_info:
            raise ValueError(f"free_id({free_id}) does not exist.")
        
        free_size = self.allocate_info.pop(free_id)
        self.usage_byte -= free_size
        self.ensure_consistent()
        return True

    def check_consistency(self):
        allocated_size = sum(self.allocate_info.values())
        return 0 <= self.usage_byte <= self.size_byte and self.usage_byte == allocated_size

    def ensure_consistent(self):
        if not self.check_consistency():
            raise RuntimeError("Memory allocation state is inconsistent.")

    def calculate_access_time_s(self, access_size_byte):
        self._validate_integer(access_size_byte, "access_size_byte", minimum = 0)
        return self.fixed_access_latency_s + access_size_byte / self.bandwidth_byte_per_s
        
class AttentionBuffer(Memory):
    def __init__(
        self,
        num_banks = 20000,
        bank_size_byte = 16000,
        banks_per_group = 32,
        read_ports_per_bank = 1,
        write_ports_per_bank = 1,
        access_width_bit = 32,
        access_latency_cycles = 3,
        clock_frequency_hz = 1000000000,
    ):
        self._validate_integer(num_banks, "num_banks", minimum = 1)
        self._validate_integer(bank_size_byte, "bank_size_byte", minimum = 1)
        self._validate_integer(banks_per_group, "banks_per_group", minimum = 1)
        self._validate_integer(read_ports_per_bank, "read_ports_per_bank", minimum = 1)
        self._validate_integer(write_ports_per_bank, "write_ports_per_bank", minimum = 1)
        self._validate_integer(access_width_bit, "access_width_bit", minimum = 1)
        self._validate_integer(access_latency_cycles, "access_latency_cycles", minimum = 0)
        self._validate_number(
            clock_frequency_hz,
            "clock_frequency_hz",
            minimum = 0,
            inclusive = False,
        )
        if access_width_bit % 8 != 0:
            raise ValueError("access_width_bit must be divisible by 8.")
        if banks_per_group > num_banks:
            raise ValueError("banks_per_group must not be greater than num_banks.")
        if num_banks % banks_per_group != 0:
            raise ValueError("num_banks must be divisible by banks_per_group.")

        size_byte = num_banks * bank_size_byte
        num_bank_groups = num_banks // banks_per_group
        bank_group_size_byte = banks_per_group * bank_size_byte
        access_width_byte = access_width_bit // 8
        bandwidth_byte_per_s = (
            num_banks
            * read_ports_per_bank
            * access_width_byte
            * clock_frequency_hz
        )
        fixed_access_latency_s = access_latency_cycles / clock_frequency_hz

        super().__init__(size_byte, bandwidth_byte_per_s, fixed_access_latency_s)

        self.num_banks = num_banks
        self.bank_size_byte = bank_size_byte
        self.banks_per_group = banks_per_group
        self.num_bank_groups = num_bank_groups
        self.bank_group_size_byte = bank_group_size_byte
        self.read_ports_per_bank = read_ports_per_bank
        self.write_ports_per_bank = write_ports_per_bank
        self.access_width_bit = access_width_bit
        self.access_width_byte = access_width_byte
        self.access_latency_cycles = access_latency_cycles
        self.clock_frequency_hz = clock_frequency_hz

        self.next_bank_group_id = 0
        self.next_bank_group_offset = [0] * num_bank_groups
        self.bank_group_usage_byte = [0] * num_bank_groups
        self.bank_usage_byte = np.zeros(num_banks, dtype=np.int64)
        self.bank_read_busy_until_cycle = [
            [0] * read_ports_per_bank for _ in range(num_banks)
        ]
        self.bank_write_busy_until_cycle = [
            [0] * write_ports_per_bank for _ in range(num_banks)
        ]
        
    def check_consistency(self):
        if np.any(
            (self.bank_usage_byte < 0)
            | (self.bank_usage_byte > self.bank_size_byte)
        ):
            return False
        if any(
            not 0 <= usage <= self.bank_group_size_byte
            for usage in self.bank_group_usage_byte
        ):
            return False

        allocated_by_bank = np.zeros(self.num_banks, dtype=np.int64)
        allocated_by_group = [0] * self.num_bank_groups
        allocated_size = 0

        for allocation in self.allocate_info.values():
            if not all(key in allocation for key in ("size", "bank_group", "bank")):
                return False

            allocation_size = allocation["size"]
            bank_group_id = allocation["bank_group"]
            bank_allocations = allocation["bank"]

            if allocation_size <= 0:
                return False
            if not 0 <= bank_group_id < self.num_bank_groups:
                return False
            if not bank_allocations:
                return False

            first_bank_id = bank_group_id * self.banks_per_group
            last_bank_id = first_bank_id + self.banks_per_group
            allocation_size_from_banks = 0

            for bank_id, allocated_size_byte in bank_allocations.items():
                if not first_bank_id <= bank_id < last_bank_id:
                    return False
                if allocated_size_byte <= 0:
                    return False

                allocated_by_bank[bank_id] += allocated_size_byte
                allocation_size_from_banks += allocated_size_byte

            if allocation_size != allocation_size_from_banks:
                return False

            allocated_by_group[bank_group_id] += allocation_size
            allocated_size += allocation_size

        if not np.array_equal(allocated_by_bank, self.bank_usage_byte):
            return False
        if allocated_by_group != self.bank_group_usage_byte:
            return False

        for bank_group_id, group_usage_byte in enumerate(self.bank_group_usage_byte):
            first_bank_id = bank_group_id * self.banks_per_group
            last_bank_id = first_bank_id + self.banks_per_group
            if group_usage_byte != np.sum(
                self.bank_usage_byte[first_bank_id:last_bank_id]
            ):
                return False

        return (
            0 <= self.usage_byte <= self.size_byte
            and self.usage_byte == allocated_size
            and self.usage_byte == np.sum(allocated_by_bank)
            and self.usage_byte == sum(allocated_by_group)
        )
    
    def allocate_memory(self, allocate_size_byte, allocate_id):
        self.ensure_consistent()
        self._validate_integer(allocate_size_byte, "allocate_size_byte", minimum = 1)
        self._validate_allocation_id(allocate_id, "allocate_id")
        
        if allocate_id in self.allocate_info:
            raise ValueError(f"allocate_id({allocate_id}) has already existed.")
        
        if self.usage_byte + allocate_size_byte > self.size_byte:
            return False
        if self.bank_group_usage_byte[self.next_bank_group_id] + allocate_size_byte > self.bank_group_size_byte:
            return False
        # Assume that allocate_size_byte % 4 == 0
        allocate_bank_num = allocate_size_byte / self.access_width_byte
        base_num_per_bank = allocate_bank_num // self.banks_per_group
        additional_num = allocate_bank_num % self.banks_per_group
        group_start_id = self.next_bank_group_id * self.banks_per_group
        group_end_id = ((self.next_bank_group_id + 1) % self.num_bank_groups) * self.banks_per_group - 1
        if not np.all(self.bank_usage_byte[group_start_id: group_end_id + 1] + self.access_width_byte * base_num_per_bank <= self.bank_size_byte):
            return False
        for i in range(additional_num):
            offset = (self.next_bank_group_offset[self.next_bank_group_id] + i) % self.banks_per_group
            if self.bank_usage_byte[group_start_id + offset] + self.access_width_byte * base_num_per_bank + self.access_width_byte > self.bank_size_byte:
                return False
        
        self.allocate_info[allocate_id] = {}
        
        self.allocate_info[allocate_id]["size"] = allocate_size_byte
        self.usage_byte += allocate_size_byte
        
        self.allocate_info[allocate_id]["bank_group"] = self.next_bank_group_id
        self.bank_group_usage_byte[self.next_bank_group_id] += allocate_size_byte
        
        self.allocate_info[allocate_id]["bank"] = {}
        self.bank_usage_byte[group_start_id: group_end_id + 1] += self.access_width_byte * base_num_per_bank
        for i in range(additional_num):
            offset = (self.next_bank_group_offset[self.next_bank_group_id] + i) % self.banks_per_group
            self.allocate_info[allocate_id]["bank"][group_start_id + offset] = self.access_width_byte
            self.bank_usage_byte[group_start_id + offset] += self.access_width_byte
        if base_num_per_bank > 0:
            for i in range(self.banks_per_group):
                if (group_start_id + i) in self.allocate_info[allocate_id]["bank"].keys():
                    self.allocate_info[allocate_id]["bank"][group_start_id + i] += self.access_width_byte * base_num_per_bank
                else:
                    self.allocate_info[allocate_id]["bank"][group_start_id + i] = self.access_width_byte * base_num_per_bank
        
        self.next_bank_group_offset[self.next_bank_group_id] = (self.next_bank_group_offset[self.next_bank_group_id] + additional_num) % self.banks_per_group
        self.next_bank_group_id = (self.next_bank_group_id + 1) % self.num_bank_groups
    
    def read(self):
        pass
    
    def write(self):
        pass
    
class HBM(Memory):
    def __init__(
        self,
        num_stacks = 8,
        stack_size_byte = 24000000000,
        bandwidth_byte_per_s = 6400000000000,
        fixed_access_latency_s = 100e-9,
    ):
        self._validate_integer(num_stacks, "num_stacks", minimum = 1)
        self._validate_integer(stack_size_byte, "stack_size_byte", minimum = 1)

        size_byte = num_stacks * stack_size_byte

        # The paper specifies HBM capacity but not bandwidth or access latency.
        # The default bandwidth (6.4 TB/s) and latency (100 ns) are assumptions.
        super().__init__(size_byte, bandwidth_byte_per_s, fixed_access_latency_s)

        self.num_stacks = num_stacks
        self.stack_size_byte = stack_size_byte
