import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402


def test_request_chip_flow():
    synthetic_trace = {
        "request_id": "synthetic-request-0",
        "input_token_num": 2,
        "output_token_num": 2,
        "arrival_cycle": 0,
    }
    request = Request(**synthetic_trace)

    config_path = Path(__file__).resolve().parent.parent / "hnlpu_config.yaml"
    config = Config(config_path)
    chip = Chip(0, 0, config)

    kv_size_per_token = (
        2
        * config.model["num_kv_heads"]
        * config.model["head_dim"]
        * config.memory["kv_dtype_bytes"]
    )

    request.status = "running"
    request.start_cycle = request.arrival_cycle

    # Prefill the two input tokens.
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
        request_cycle = request.arrival_cycle,
    )

    prefill_attention_task = ComputeTask(
        request_id = request.request_id,
        task_type = "attention",
        workload = {
            "q_length": 2,
            "kv_length": 2,
        },
    )
    prefill_attention_result = chip.vex.execute(
        prefill_attention_task,
        request_cycle = prefill_projection_result["finish_cycle"],
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
    assert prefill_store_result is not False

    prefill_ab_usage_byte = chip.attention_buffer.usage_byte
    assert prefill_projection_result["request_cycle"] == request.arrival_cycle
    assert (
        prefill_attention_result["request_cycle"]
        == prefill_projection_result["finish_cycle"]
    )
    assert (
        prefill_store_result["request_cycle"]
        == prefill_attention_result["finish_cycle"]
    )
    assert prefill_kv_block.storage_location == "attention_buffer"
    assert prefill_kv_block.allocate_id in chip.attention_buffer.allocate_info
    assert prefill_kv_block.allocate_id not in chip.hbm.allocate_info
    assert chip.hbm.usage_byte == 0
    assert prefill_ab_usage_byte == 2 * kv_size_per_token

    # Token 3 is the abstract output of prefill and becomes the decode input.
    request.generated_token_num = 1
    request.current_token_position = 3
    request.phase = "decode"

    decode_projection_task = ComputeTask(
        request_id = request.request_id,
        task_type = "projection",
        workload = {
            "projection_type": "qkv",
            "num_tokens": 1,
        },
    )
    decode_projection_result = chip.hn_array.execute(
        decode_projection_task,
        request_cycle = prefill_store_result["finish_cycle"],
    )

    # Only token 1 and token 2 are read from the historical KV cache.
    decode_read_result = chip.kv_cache_manager.read_kv_blocks(
        [prefill_kv_block.block_id],
        request_cycle = decode_projection_result["finish_cycle"],
    )
    assert decode_read_result["attention_buffer_result"] is not None
    assert decode_read_result["hbm_result"] is None
    assert decode_read_result["total_read_size_byte"] == 2 * kv_size_per_token
    assert "synthetic-request-0-token-3-kv" not in chip.kv_cache_manager.kv_cache_blocks

    decode_attention_task = ComputeTask(
        request_id = request.request_id,
        task_type = "attention",
        workload = {
            "q_length": 1,
            "kv_length": 3,
        },
    )
    decode_attention_result = chip.vex.execute(
        decode_attention_task,
        request_cycle = decode_read_result["finish_cycle"],
    )

    token_3_kv_block = KVcacheBlock(
        block_id = "synthetic-request-0-token-3-kv",
        request_id = request.request_id,
        layer_id = 0,
        first_token_position = 3,
        num_tokens = 1,
        size_byte = kv_size_per_token,
        chip_id = chip.chip_id,
    )
    decode_store_result = chip.kv_cache_manager.store_kv_block(
        token_3_kv_block,
        request_cycle = decode_attention_result["finish_cycle"],
    )
    assert decode_store_result is not False

    assert (
        decode_projection_result["request_cycle"]
        == prefill_store_result["finish_cycle"]
    )
    assert (
        decode_read_result["request_cycle"]
        == decode_projection_result["finish_cycle"]
    )
    assert (
        decode_attention_result["request_cycle"]
        == decode_read_result["finish_cycle"]
    )
    assert (
        decode_store_result["request_cycle"]
        == decode_attention_result["finish_cycle"]
    )
    assert decode_attention_result["compute_workload"]["q_length"] == 1
    assert decode_attention_result["compute_workload"]["kv_length"] == 3
    assert token_3_kv_block.storage_location == "attention_buffer"
    assert chip.hbm.usage_byte == 0
    assert chip.attention_buffer.usage_byte == 3 * kv_size_per_token
    assert chip.kv_cache_manager.check_consistency()

    # Token 4 is the abstract output of the decode step.
    request.generated_token_num = 2
    request.current_token_position = 4
    request.status = "finished"
    request.finish_cycle = decode_store_result["finish_cycle"]

    assert request.generated_token_num == request.output_token_num
    assert request.current_token_position == 4
    assert request.phase == "decode"
    assert request.status == "finished"
    assert request.start_cycle == request.arrival_cycle
    assert request.finish_cycle == decode_store_result["finish_cycle"]

    print(
        "Prefill cycles: "
        f"projection={prefill_projection_result['start_cycle']}-"
        f"{prefill_projection_result['finish_cycle']}, "
        f"attention={prefill_attention_result['start_cycle']}-"
        f"{prefill_attention_result['finish_cycle']}, "
        f"kv_store={prefill_store_result['start_cycle']}-"
        f"{prefill_store_result['finish_cycle']}"
    )
    print(
        "Decode cycles: "
        f"projection={decode_projection_result['start_cycle']}-"
        f"{decode_projection_result['finish_cycle']}, "
        f"history_kv_read={decode_read_result['start_cycle']}-"
        f"{decode_read_result['finish_cycle']}, "
        f"attention={decode_attention_result['start_cycle']}-"
        f"{decode_attention_result['finish_cycle']}, "
        f"kv_store={decode_store_result['start_cycle']}-"
        f"{decode_store_result['finish_cycle']}"
    )
    print(
        "Attention Buffer usage: "
        f"after_prefill={prefill_ab_usage_byte} bytes, "
        f"after_decode={chip.attention_buffer.usage_byte} bytes"
    )
    print(
        "Final Request state: "
        f"generated_token_num={request.generated_token_num}, "
        f"current_token_position={request.current_token_position}, "
        f"phase={request.phase}, status={request.status}, "
        f"finish_cycle={request.finish_cycle}"
    )


if __name__ == "__main__":
    test_request_chip_flow()
