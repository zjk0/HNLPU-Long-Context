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
        # Validate the block, request cycle, chip ownership, and block state.
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

        # Ensure all values used as dictionary keys are hashable.
        for value, name in (
            (block.block_id, "block_id"),
            (block.request_id, "request_id"),
            (block.layer_id, "layer_id"),
        ):
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be hashable.") from exc

        # Detect an allocation that exists in memory but is missing from Manager.
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

        # Prefer the Attention Buffer and fall back to HBM when it cannot fit.
        if self.attention_buffer.allocate_memory(block.size_byte,block.block_id,):
            target_memory = self.attention_buffer
            storage_location = "attention_buffer"
        elif self.hbm.allocate_memory(block.size_byte, block.block_id):
            target_memory = self.hbm
            storage_location = "hbm"
        else:
            return False

        # Write the block and release its allocation if writing raises an error.
        try:
            write_result = target_memory.write(block.block_id,request_cycle,)
        except Exception:
            target_memory.free_memory(block.block_id)
            raise

        # Treat an explicit write failure as a failed store and roll back.
        if write_result is False:
            target_memory.free_memory(block.block_id)
            return False

        # Use request and layer together to index blocks for attention reads.
        request_layer_key = (block.request_id, block.layer_id)

        # Publish placement metadata and all Manager indexes after writing.
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
            # Undo every published state change before releasing the allocation.
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

        # Return the timing information produced by the selected memory.
        return write_result

    def read_kv_blocks(self, block_ids, request_cycle):
        # Validate the request parameters.
        if not isinstance(block_ids, list):
            raise TypeError("block_ids must be a list.")
        if not block_ids:
            raise ValueError("block_ids must not be empty.")
        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")

        seen_block_ids = set()
        seen_allocate_ids = set()

        for block_id in block_ids:
            if block_id is None or (isinstance(block_id, str) and not block_id.strip()):
                raise ValueError("block_id must not be empty.")
            try:
                hash(block_id)
            except TypeError as exc:
                raise TypeError("block_id must be hashable.") from exc

            if block_id in seen_block_ids:
                raise ValueError(f"block_id({block_id}) is duplicated in block_ids.")
            if block_id not in self.kv_cache_blocks:
                raise ValueError(f"block_id({block_id}) does not exist.")

            seen_block_ids.add(block_id)
            block = self.kv_cache_blocks[block_id]

            # Validate Manager ownership and index consistency.
            if block.block_id != block_id:
                raise RuntimeError(f"block_id({block_id}) does not match its block record.")
            if block.chip_id != self.chip_id:
                raise RuntimeError(f"block_id({block_id}) does not belong to this chip.")
            if block_id not in self.request_blocks.get(block.request_id, set()):
                raise RuntimeError(f"block_id({block_id}) is missing from request_blocks.")

            request_layer_key = (block.request_id, block.layer_id)
            if block_id not in self.request_layer_blocks.get(request_layer_key, []):
                raise RuntimeError(f"block_id({block_id}) is missing from request_layer_blocks.")

            # Validate placement metadata and the corresponding allocation.
            if block.storage_location not in ("attention_buffer", "hbm"):
                raise RuntimeError(f"block_id({block_id}) has an invalid storage_location.")
            if block.allocate_id is None:
                raise RuntimeError(f"block_id({block_id}) does not have an allocate_id.")
            try:
                hash(block.allocate_id)
            except TypeError as exc:
                raise RuntimeError(f"allocate_id of block_id({block_id}) must be hashable.") from exc
            if block.allocate_id in seen_allocate_ids:
                raise RuntimeError(f"allocate_id({block.allocate_id}) is shared by multiple blocks.")

            seen_allocate_ids.add(block.allocate_id)
            if block.storage_location == "attention_buffer":
                target_memory = self.attention_buffer
                other_memory = self.hbm
            else:
                target_memory = self.hbm
                other_memory = self.attention_buffer

            if block.allocate_id not in target_memory.allocate_info:
                raise RuntimeError(
                    f"allocate_id({block.allocate_id}) of block_id"
                    f"({block_id}) does not exist in its target memory."
                )
            if block.allocate_id in other_memory.allocate_info:
                raise RuntimeError(
                    f"allocate_id({block.allocate_id}) of block_id"
                    f"({block_id}) exists in both memories."
                )
            if "ready_cycle" not in target_memory.allocate_info[block.allocate_id]:
                raise RuntimeError(f"block_id({block_id}) has not completed its write.")

        # Memory grouping and read operations
        allocate_ids_attention_buffer = []
        allocate_ids_hbm = []
        for block_id in block_ids:
            block = self.kv_cache_blocks[block_id]
            if block.storage_location == "attention_buffer":
                allocate_ids_attention_buffer.append(block.allocate_id)
            elif block.storage_location == "hbm":
                allocate_ids_hbm.append(block.allocate_id)

        if allocate_ids_attention_buffer and not allocate_ids_hbm:
            read_result_attention_buffer = self.attention_buffer.read(
                allocate_ids_attention_buffer,
                request_cycle,
            )
            return read_result_attention_buffer
        elif not allocate_ids_attention_buffer and allocate_ids_hbm:
            read_result_hbm = self.hbm.read(
                allocate_ids_hbm,
                request_cycle,
            )
            return read_result_hbm
        elif allocate_ids_attention_buffer and allocate_ids_hbm:
            read_result_attention_buffer = self.attention_buffer.read(
                allocate_ids_attention_buffer,
                request_cycle,
            )
            read_result_hbm = self.hbm.read(
                allocate_ids_hbm,
                request_cycle,
            )

            start_cycle = min(
                read_result_attention_buffer["start_cycle"],
                read_result_hbm["start_cycle"],
            )
            finish_cycle = max(
                read_result_attention_buffer["finish_cycle"],
                read_result_hbm["finish_cycle"],
            )

            return {
                "request_cycle": request_cycle,
                "start_cycle": start_cycle,
                "finish_cycle": finish_cycle,
                "wait_cycles": start_cycle - request_cycle,
                "service_cycles": finish_cycle - start_cycle,
                "total_latency_cycles": finish_cycle - request_cycle,
                "total_read_size_byte": (
                    read_result_attention_buffer["total_read_size_byte"]
                    + read_result_hbm["total_read_size_byte"]
                ),
                "bank_read_size": read_result_attention_buffer[
                    "bank_read_size"
                ],
                "bank_read_issue_cycles": read_result_attention_buffer[
                    "bank_read_issue_cycles"
                ],
                "hbm_transfer_cycles": read_result_hbm[
                    "transfer_cycles"
                ],
                "attention_buffer_result": read_result_attention_buffer,
                "hbm_result": read_result_hbm,
            }

        raise RuntimeError("No valid KV cache allocation was found to read.")


    def free_kv_block(self, block_id):
        pass

    def free_request(self, request_id):
        pass
