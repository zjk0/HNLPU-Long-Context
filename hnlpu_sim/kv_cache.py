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
        self.token_stride = 4

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

    def store_kv_block(self, block: KVcacheBlock, request_cycle):
        if not isinstance(block, KVcacheBlock):
            raise TypeError("block must be a KVcacheBlock.")
        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")
        if block.chip_id != self.chip_id:
            return False
        if block.block_id in self.kv_cache_blocks:
            return False
        if block.storage_location is not None or block.allocate_id is not None:
            return False

        for value, name in (
            (block.block_id, "block_id"),
            (block.request_id, "request_id"),
            (block.layer_id, "layer_id"),
        ):
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be hashable.") from exc

        if (
            block.block_id in self.attention_buffer.allocate_info
            or block.block_id in self.hbm.allocate_info
        ):
            raise RuntimeError(
                f"block_id({block.block_id}) already exists in memory "
                "but is not registered in KVcacheManager."
            )

        target_memory = None
        storage_location = None

        if self.attention_buffer.allocate_memory(block.size_byte,block.block_id,):
            target_memory = self.attention_buffer
            storage_location = "attention_buffer"
        elif self.hbm.allocate_memory(block.size_byte, block.block_id):
            target_memory = self.hbm
            storage_location = "hbm"
        else:
            return False

        try:
            write_result = target_memory.write(block.block_id,request_cycle,)
        except Exception:
            target_memory.free_memory(block.block_id)
            raise

        if write_result is False:
            target_memory.free_memory(block.block_id)
            return False

        request_layer_key = (block.request_id, block.layer_id)

        try:
            block.storage_location = storage_location
            block.allocate_id = block.block_id

            self.kv_cache_blocks[block.block_id] = block
            self.request_blocks.setdefault(block.request_id, set()).add(block.block_id)
            layer_block_ids = self.request_layer_blocks.setdefault(request_layer_key, [])
            layer_block_ids.append(block.block_id)
            layer_block_ids.sort(
                key = lambda block_id: self.kv_cache_blocks[block_id].first_token_position
            )
        except Exception:
            self.kv_cache_blocks.pop(block.block_id, None)

            request_block_ids = self.request_blocks.get(block.request_id)
            if request_block_ids is not None:
                request_block_ids.discard(block.block_id)
                if not request_block_ids:
                    self.request_blocks.pop(block.request_id)

            layer_block_ids = self.request_layer_blocks.get(request_layer_key)
            if layer_block_ids is not None and block.block_id in layer_block_ids:
                layer_block_ids.remove(block.block_id)
                if not layer_block_ids:
                    self.request_layer_blocks.pop(request_layer_key)

            block.storage_location = None
            block.allocate_id = None
            target_memory.free_memory(block.block_id)
            raise

        return write_result

    def read_kv_blocks(self, block_ids):
        pass

    def free_kv_block(self, block_id):
        pass
