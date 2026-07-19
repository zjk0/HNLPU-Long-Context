class Memory:
    def __init__(self, size_byte, bandwidth_byte_per_s, fixed_access_latency_s):
        self._validate_integer(size_byte, "size_byte", minimum = 1)
        self._validate_number(bandwidth_byte_per_s, "bandwidth_byte_per_s", minimum = 0, inclusive = False)
        self._validate_number(fixed_access_latency_s, "fixed_access_latency_s", minimum = 0)

        self.size_byte = size_byte
        self.bandwidth_byte_per_s = bandwidth_byte_per_s
        self.fixed_access_latency_s = fixed_access_latency_s
        self._usage_byte = 0
        self._allocate_info = {}

    @property
    def usage_byte(self):
        return self._usage_byte

    @property
    def allocate_info(self):
        return self._allocate_info.copy()

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
        self._ensure_consistent()
        self._validate_integer(allocate_size_byte, "allocate_size_byte", minimum = 1)
        self._validate_allocation_id(allocate_id, "allocate_id")
        
        # Allocate ID can not repeat.
        if allocate_id in self._allocate_info:
            raise ValueError(f"allocate_id({allocate_id}) has already existed.")
        
        if self.usage_byte + allocate_size_byte > self.size_byte:
            return False
        
        self._usage_byte += allocate_size_byte
        self._allocate_info[allocate_id] = allocate_size_byte
        self._ensure_consistent()
        return True
        
    def free_memory(self, free_id):
        self._ensure_consistent()
        self._validate_allocation_id(free_id, "free_id")

        # The ID to free must exist
        if free_id not in self._allocate_info:
            raise ValueError(f"free_id({free_id}) does not exist.")
        
        free_size = self._allocate_info.pop(free_id)
        self._usage_byte -= free_size
        self._ensure_consistent()
        return True

    def check_consistency(self):
        allocated_size = sum(self._allocate_info.values())
        return 0 <= self.usage_byte <= self.size_byte and self.usage_byte == allocated_size

    def _ensure_consistent(self):
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
        read_ports_per_bank = 1,
        write_ports_per_bank = 1,
        access_width_bit = 32,
        access_latency_cycles = 3,
        clock_frequency_hz = 1000000000,
    ):
        self._validate_integer(num_banks, "num_banks", minimum = 1)
        self._validate_integer(bank_size_byte, "bank_size_byte", minimum = 1)
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

        size_byte = num_banks * bank_size_byte
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
        self.read_ports_per_bank = read_ports_per_bank
        self.write_ports_per_bank = write_ports_per_bank
        self.access_width_bit = access_width_bit
        self.access_width_byte = access_width_byte
        self.access_latency_cycles = access_latency_cycles
        self.clock_frequency_hz = clock_frequency_hz
        
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
