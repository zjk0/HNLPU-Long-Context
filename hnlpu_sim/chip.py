from memory import AttentionBuffer, HBM
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
        pass
