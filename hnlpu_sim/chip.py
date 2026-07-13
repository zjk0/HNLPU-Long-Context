from memory import Memory
from kv_cache import KVcacheBlock, KVcacheManager

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
        
    