class Memory:
    def __init__(self, size_byte, bandwidth_byte_per_s, fixed_access_latency_s):
        if not isinstance(size_byte, int) or isinstance(size_byte, bool) or size_byte <= 0:
            raise ValueError("size_byte must be a positive integer.")
        if (
            not isinstance(bandwidth_byte_per_s, (int, float))
            or isinstance(bandwidth_byte_per_s, bool)
            or bandwidth_byte_per_s <= 0
        ):
            raise ValueError("bandwidth_byte_per_s must be positive.")
        if (
            not isinstance(fixed_access_latency_s, (int, float))
            or isinstance(fixed_access_latency_s, bool)
            or fixed_access_latency_s < 0
        ):
            raise ValueError("fixed_access_latency_s must be non-negative.")

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
    
    def get_usage(self):
        return self.usage_byte
    
    def get_remain_size(self):
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
        if (
            not isinstance(allocate_size_byte, int)
            or isinstance(allocate_size_byte, bool)
            or allocate_size_byte <= 0
        ):
            raise ValueError("allocate_size_byte must be a positive integer.")
        if allocate_id is None or (isinstance(allocate_id, str) and not allocate_id.strip()):
            raise ValueError("allocate_id must not be empty.")
        try:
            hash(allocate_id)
        except TypeError as exc:
            raise TypeError("allocate_id must be hashable.") from exc
        
        # Allocate ID can not repeat.
        if allocate_id in self._allocate_info:
            raise ValueError(f"allocate_id({allocate_id}) has already existed.")
        
        if self.usage_byte + allocate_size_byte > self.size_byte:
            return False
        
        self._usage_byte += allocate_size_byte
        self._allocate_info[allocate_id] = allocate_size_byte
        if not self.check_consistency():
            raise RuntimeError("Memory allocation state is inconsistent.")
        return True
        
    def free_memory(self, free_id):
        # The ID to free must exist
        if free_id not in self._allocate_info:
            raise ValueError(f"The free_id({free_id}) does not exist.")
        
        free_size = self._allocate_info.pop(free_id)
        self._usage_byte -= free_size
        if not self.check_consistency():
            raise RuntimeError("Memory allocation state is inconsistent.")
        return True

    def check_consistency(self):
        allocated_size = sum(self._allocate_info.values())
        return 0 <= self.usage_byte <= self.size_byte and self.usage_byte == allocated_size

    def calculate_access_time_s(self, access_size_byte):
        if (
            not isinstance(access_size_byte, int)
            or isinstance(access_size_byte, bool)
            or access_size_byte < 0
        ):
            raise ValueError("access_size_byte must be a non-negative integer.")

        return self.fixed_access_latency_s + access_size_byte / self.bandwidth_byte_per_s
        
class AttentionBuffer(Memory):
    def __init__(self):
        pass
    
class HBM(Memory):
    def __init__(self):
        pass
