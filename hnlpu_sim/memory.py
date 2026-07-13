class Memory:
    def __init__(self, memory_type, size_byte, bandwidth_byte_per_s, fixed_latency_s):
        self.memory_type = memory_type
        self.size_byte = size_byte
        self.bandwidth_byte_per_s = bandwidth_byte_per_s
        self.fixed_latency_s = fixed_latency_s
        self.usage_byte = 0
        
    def get_total_size(self):
        return self.size_byte
    
    def get_usage(self):
        return self.usage_byte
    
    def get_remain_size(self):
        return self.size_byte - self.usage_byte
        
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
        
    def can_allocate(self, allocate_size_byte):
        if self.usage_byte + allocate_size_byte > self.size_byte:
            return False
        else:
            return True
        
    def can_free(self, free_size_byte):
        if self.usage_byte - free_size_byte < 0:
            return False
        else:
            return True
        
    def allocate_memory(self, allocate_size_byte):
        if self.can_allocate(allocate_size_byte):
            self.usage_byte += allocate_size_byte
            return True
        else:
            self.usage_byte = self.size_byte
            return False
        
    def free_memory(self, free_size_byte):
        if self.can_free(free_size_byte):
            self.usage_byte -= free_size_byte
            return True
        else:
            self.usage_byte = 0
            return False