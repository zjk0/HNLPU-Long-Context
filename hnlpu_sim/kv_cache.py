import uuid
import numpy as np
from memory import AttentionBuffer, HBM

class KVcacheBlock:
    def __init__(
        self,
        block_id,
        request_id,
        layer_id,
        first_token_position,
        num_tokens,
        size_byte,
        chip_id = None,
        storage_location = None,
        allocate_id = None,
    ):
        # Block identity and covered token range.
        self.block_id = block_id
        self.request_id = request_id
        self.layer_id = layer_id
        self.first_token_position = first_token_position
        self.num_tokens = num_tokens
        self.size_byte = size_byte

        # Runtime placement information, assigned by KVcacheManager.
        self.chip_id = chip_id
        self.storage_location = storage_location
        self.allocate_id = allocate_id
    
class KVcacheManager:
    def __init__(
        self, 
        chip_id,  
        attention_buffer: AttentionBuffer, 
        hbm: HBM
    ):
        self.chip_id = chip_id
        self.attention_buffer = attention_buffer
        self.hbm = hbm
        self.kv_cache_blocks = {}
        self.request_blocks = {}
        self.request_layer_blocks = {}
