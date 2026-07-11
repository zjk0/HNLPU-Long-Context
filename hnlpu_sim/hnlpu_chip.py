class Memory:
    def __init__(self, size_mb, bandwidth_gb_per_s):
        self.size_mb = size_mb
        self.bandwidth_gb_per_s = bandwidth_gb_per_s
        self.usage_mb = 0
        
    def is_full(self):
        if self.usage_mb >= self.size_mb:
            return True
        else:
            return False
        
    def is_empty(self):
        if self.usage_mb == 0:
            return True
        else:
            return False
        
    def can_allocate(self, allocate_size_mb):
        if self.usage_mb + allocate_size_mb > self.size_mb:
            return False
        else:
            return True
        
    def can_free(self, free_size_mb):
        if self.usage_mb - free_size_mb < 0:
            return False
        else:
            return True
        
    def allocate_memory(self, allocate_size_mb):
        if self.can_allocate(allocate_size_mb):
            self.usage_mb += allocate_size_mb
            return True
        else:
            self.usage_mb = self.size_mb
            return False
        
    def free_memory(self, free_size_mb):
        if self.can_free(free_size_mb):
            self.usage_mb -= free_size_mb
            return True
        else:
            self.usage_mb = 0
            return False

class Chip:
    def __init__(
        self, 
        row, 
        column, 
        attention_buffer_size_mb = 320, 
        attention_buffer_bandwidth_gb_per_s = 80000, 
        hbm_size_mb = 196608, 
        hbm_bandwidth_gb_per_s = 6400
    ):
        self.row = row
        self.column = column
        self.attention_buffer = Memory(attention_buffer_size_mb, attention_buffer_bandwidth_gb_per_s)
        self.hbm = Memory(hbm_size_mb, hbm_bandwidth_gb_per_s)
        self.current_tasks = []
        
    def get_current_tasks(self):
        return self.current_tasks
    
    def is_full(self, memory_type):
        if memory_type == "attention_buffer":
            return self.attention_buffer.is_full()
        elif memory_type == "hbm":
            return self.hbm.is_full()
        else:
            raise ValueError("Invalid memory type. Please use attention_buffer or hbm.")
        
    def is_empty(self, memory_type):
        if memory_type == "attention_buffer":
            return self.attention_buffer.is_empty()
        elif memory_type == "hbm":
            return self.hbm.is_empty()
        else:
            raise ValueError("Invalid memory type. Please use attention_buffer or hbm.")
        
    def allocate(self, allocate_size_mb):
        usage = self.attention_buffer.usage_mb
        if self.attention_buffer.allocate_memory(allocate_size_mb):
            return True
        else:
            if self.hbm.allocate_memory(allocate_size_mb - (self.attention_buffer.size_mb - usage)):
                return True
            else:
                return False
            
    def free(self, free_size_mb):
        pass
        
    