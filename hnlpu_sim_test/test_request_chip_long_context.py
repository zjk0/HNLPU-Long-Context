import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402

def test():
    test_trace = {
        "request_id": "synthetic-request-0",
        "input_token_num": 2,
        "output_token_num": 50,
        "arrival_cycle": 0,
    }
    request = Request(**test_trace)
    
    config_path = Path(__file__).resolve().parent.parent / "hnlpu_config.yaml"
    config = Config(
        config_path,
        overrides={
            "memory": {
                "attention_buffer_mb_per_chip": 0.064,
                "attention_buffer_banks_per_chip": 32,
                "attention_buffer_bank_kb": 2,
            }
        },
    )
    chip = Chip(0, 0, config)
    
    kv_size_per_token = 2 * config.model["num_kv_heads"] * config.model["head_dim"] * config.memory["kv_dtype_bytes"]
    
    request.status = "running"
    request.phase = "prefill"
    request.start_cycle = request.arrival_cycle
    
    prefill_projection_task = ComputeTask(
        request_id = request.request_id,
        task_type = "projection",
        workload = {
            "projection_type": "qkv",
            "num_tokens": request.input_token_num,
        },
    )
    prefill_projection_result = chip.hn_array.execute(
        prefill_projection_task,
        request_cycle = request.arrival_cycle
    )
    
    prefill_attention_task = ComputeTask(
        request_id = request.request_id,
        task_type = "attention",
        workload = {
            "q_length": 2,
            "kv_length": 2,
        }
    )
    prefill_attention_result = chip.vex.execute(
        prefill_attention_task, 
        request_cycle = prefill_projection_result["finish_cycle"]
    )
    
    prefill_kv_block = KVcacheBlock(
        block_id = "synthetic-request-0-prefill-kv",
        request_id = request.request_id,
        layer_id = 0,
        first_token_position = 1,
        num_tokens = 2,
        size_byte = 2 * kv_size_per_token,
        chip_id = chip.chip_id,
    )
    
    prefill_store_result = chip.kv_cache_manager.store_kv_block(
        prefill_kv_block,
        request_cycle = prefill_attention_result["finish_cycle"],
    )
    
    request.generated_token_num = 1
    request.phase = "decode"
    request.current_token_position = 3
    kv_cache_ids = [prefill_kv_block.block_id]
    previous_store_result = prefill_store_result
    
    for i in range(request.output_token_num - 1):
        decode_projection_task = ComputeTask(
            request_id = request.request_id, 
            task_type = "projection", 
            workload = {
                "projection_type": "qkv", 
                "num_tokens": 1
            }
        )
        decode_projection_result = chip.hn_array.execute(
            decode_projection_task,
            request_cycle = previous_store_result["finish_cycle"],
        )
        
        decode_read_result = chip.kv_cache_manager.read_kv_blocks(
            kv_cache_ids, 
            request_cycle = decode_projection_result["finish_cycle"]
        )
        
        decode_attention_task = ComputeTask(
            request_id = request.request_id, 
            task_type = "attention", 
            workload = {
                "q_length": 1, 
                "kv_length": (i + 3)
            }
        )
        decode_attention_result = chip.vex.execute(
            decode_attention_task, 
            request_cycle = decode_read_result["finish_cycle"]
        )
        
        new_token_block = KVcacheBlock(
            block_id = f"synthetic-request-0-decode-kv-{i + 3}",
            request_id = request.request_id,
            layer_id = 0,
            first_token_position = i + 3,
            num_tokens = 1,
            size_byte = kv_size_per_token,
            chip_id = chip.chip_id
        )
        new_token_block_store_result = chip.kv_cache_manager.store_kv_block(
            new_token_block, 
            request_cycle = decode_attention_result["finish_cycle"]
        )
        previous_store_result = new_token_block_store_result
        
        request.generated_token_num += 1
        request.current_token_position += 1
        kv_cache_ids.append(new_token_block.block_id)
        
    request.status = "finished"
    request.finish_cycle = new_token_block_store_result["finish_cycle"]
    
if __name__ == "__main__":
    test()
