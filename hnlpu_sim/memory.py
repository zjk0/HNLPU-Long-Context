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