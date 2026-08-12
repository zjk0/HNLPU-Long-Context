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
    config = Config(config_path)
    chip = Chip(0, 0, config)
    
    kv_cache_size_per_token = 2 * config.model["num_kv_heads"] * config.model["head_dim"] * config.memory["kv_dtype_bytes"]
    
    request.status = "prefill"
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
    
if __name__ == "__main__":
    test()