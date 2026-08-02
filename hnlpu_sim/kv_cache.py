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
        hbm: HBM,
        check_consistency = False,
    ):
        if not isinstance(check_consistency, bool):
            raise TypeError("check_consistency must be a boolean.")

        self.chip_id = chip_id
        self.attention_buffer = attention_buffer
        self.hbm = hbm
        self.consistency_check_enabled = check_consistency
        self.kv_cache_blocks = {}
        self.request_blocks = {}
        self.request_layer_blocks = {}

    def check_consistency(self):
        try:
            if not self.attention_buffer.check_consistency():
                return False
            if not self.hbm.check_consistency():
                return False

            expected_request_blocks = {}
            expected_request_layer_blocks = {}
            seen_allocate_ids = set()

            for block_id, block in self.kv_cache_blocks.items():
                if not isinstance(block, KVcacheBlock):
                    return False
                if block.block_id != block_id:
                    return False
                if block.chip_id != self.chip_id:
                    return False
                if block.storage_location not in ("attention_buffer", "hbm"):
                    return False
                if block.allocate_id is None:
                    return False

                hash(block.request_id)
                hash(block.layer_id)
                hash(block.allocate_id)

                if block.allocate_id in seen_allocate_ids:
                    return False
                seen_allocate_ids.add(block.allocate_id)

                if block.storage_location == "attention_buffer":
                    target_memory = self.attention_buffer
                    other_memory = self.hbm
                else:
                    target_memory = self.hbm
                    other_memory = self.attention_buffer

                if block.allocate_id not in target_memory.allocate_info:
                    return False
                if block.allocate_id in other_memory.allocate_info:
                    return False

                allocation = target_memory.allocate_info[block.allocate_id]
                if allocation.get("size") != block.size_byte:
                    return False
                if "ready_cycle" not in allocation:
                    return False

                expected_request_blocks.setdefault(block.request_id, set()).add(block_id)
                expected_request_layer_blocks.setdefault((block.request_id, block.layer_id), []).append(block_id)

            for block_ids in expected_request_layer_blocks.values():
                block_ids.sort(
                    key = lambda block_id: self.kv_cache_blocks[block_id].first_token_position
                )

            return (
                self.request_blocks == expected_request_blocks
                and self.request_layer_blocks
                == expected_request_layer_blocks
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def ensure_consistent(self):
        if self.consistency_check_enabled and not self.check_consistency():
            raise RuntimeError("KVcacheManager state is inconsistent.")

    def store_kv_block(self, block: KVcacheBlock, request_cycle):
        self.ensure_consistent()

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
        if block.block_id in self.attention_buffer.allocate_info or block.block_id in self.hbm.allocate_info:
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

        self.ensure_consistent()

        # Return the timing information produced by the selected memory.
        return write_result

    def read_kv_blocks(self, block_ids, request_cycle):
        self.ensure_consistent()

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

        # Memory grouping and read operations
        allocate_ids_attention_buffer = []
        allocate_ids_hbm = []
        for block_id in block_ids:
            block = self.kv_cache_blocks[block_id]
            if block.storage_location == "attention_buffer":
                allocate_ids_attention_buffer.append(block.allocate_id)
            elif block.storage_location == "hbm":
                allocate_ids_hbm.append(block.allocate_id)

        read_result_attention_buffer = None
        read_result_hbm = None

        if allocate_ids_attention_buffer:
            read_result_attention_buffer = self.attention_buffer.read(allocate_ids_attention_buffer, request_cycle)
        if allocate_ids_hbm:
            read_result_hbm = self.hbm.read(allocate_ids_hbm, request_cycle)

        read_results = [
            read_result for read_result in (read_result_attention_buffer, read_result_hbm)
            if read_result is not None
        ]
        if not read_results:
            raise RuntimeError("No valid KV cache allocation was found to read.")

        start_cycle = min(read_result["start_cycle"] for read_result in read_results)
        finish_cycle = max(read_result["finish_cycle"] for read_result in read_results)

        return {
            "request_cycle": request_cycle,
            "start_cycle": start_cycle,
            "finish_cycle": finish_cycle,
            "wait_cycles": start_cycle - request_cycle,
            "service_cycles": finish_cycle - start_cycle,
            "total_latency_cycles": finish_cycle - request_cycle,
            "total_read_size_byte": sum(read_result["total_read_size_byte"] for read_result in read_results),
            "attention_buffer_result": read_result_attention_buffer,
            "hbm_result": read_result_hbm,
        }

    def free_kv_block(self, block_id):
        self.ensure_consistent()

        if block_id not in self.kv_cache_blocks:
            raise ValueError(f"block_id({block_id}) does not exist.")

        block = self.kv_cache_blocks[block_id]
        if block_id not in self.request_blocks.get(block.request_id, set()):
            raise RuntimeError(f"block_id({block_id}) is missing from request_blocks.")
        if block_id not in self.request_layer_blocks.get((block.request_id, block.layer_id), []):
            raise RuntimeError(f"block_id({block_id}) is missing from request_layer_blocks.")

        if block.storage_location == "attention_buffer":
            free_status = self.attention_buffer.free_memory(block.allocate_id)
        elif block.storage_location == "hbm":
            free_status = self.hbm.free_memory(block.allocate_id)
        else:
            return False

        if free_status == False:
            return False
        else:
            self.kv_cache_blocks.pop(block_id, None)

            request_block_ids = self.request_blocks[block.request_id]
            request_block_ids.discard(block_id)
            if not request_block_ids:
                self.request_blocks.pop(block.request_id)

            request_layer_key = (block.request_id, block.layer_id)
            layer_block_ids = self.request_layer_blocks[request_layer_key]
            layer_block_ids.remove(block_id)
            if not layer_block_ids:
                self.request_layer_blocks.pop(request_layer_key)

        self.ensure_consistent()
        return True

    def free_request(self, request_id):
        self.ensure_consistent()

        if request_id not in self.request_blocks:
            raise ValueError(f"request_id({request_id}) does not exist.")

        # Copy the IDs because free_kv_block() updates request_blocks.
        block_ids = list(self.request_blocks[request_id])
        for block_id in block_ids:
            if self.free_kv_block(block_id) is False:
                return False

        self.ensure_consistent()
        return True
