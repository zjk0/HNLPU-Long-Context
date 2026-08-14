import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402

def test_request_chip_long_context_hbm_overflow():
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

    kv_size_per_token = (
        2
        * config.model["num_kv_heads"]
        * config.model["head_dim"]
        * config.memory["kv_dtype_bytes"]
    )
    assert chip.attention_buffer.size_byte == 64_000
    assert kv_size_per_token == 2_048

    request.status = "running"
    request.phase = "prefill"
    request.start_cycle = request.arrival_cycle

    # Prefill two input tokens and store their KV cache in the Attention Buffer.
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
    assert chip.attention_buffer.usage_byte == 2 * kv_size_per_token
    assert chip.hbm.usage_byte == 0

    # Token 3 is produced by prefill and becomes the first decode input.
    request.generated_token_num = 1
    request.phase = "decode"
    request.current_token_position = 3
    kv_cache_ids = [prefill_kv_block.block_id]
    previous_store_result = prefill_store_result

    saw_attention_buffer_only_read = False
    saw_mixed_read = False
    first_hbm_token_position = None
    first_decode_start_cycle = None

    for i in range(request.output_token_num - 1):
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
            request_cycle = previous_store_result["finish_cycle"],
        )
        if first_decode_start_cycle is None:
            first_decode_start_cycle = decode_projection_result["start_cycle"]
        assert (
            decode_projection_result["request_cycle"]
            == previous_store_result["finish_cycle"]
        )

        # Read only historical KV. The current token's K/V comes from projection.
        decode_read_result = chip.kv_cache_manager.read_kv_blocks(
            kv_cache_ids,
            request_cycle = decode_projection_result["finish_cycle"],
        )
        assert (
            decode_read_result["request_cycle"]
            == decode_projection_result["finish_cycle"]
        )
        assert decode_read_result["total_read_size_byte"] == (
            i + request.input_token_num
        ) * kv_size_per_token

        attention_buffer_result = decode_read_result["attention_buffer_result"]
        hbm_result = decode_read_result["hbm_result"]
        if attention_buffer_result is not None and hbm_result is None:
            saw_attention_buffer_only_read = True
        if attention_buffer_result is not None and hbm_result is not None:
            saw_mixed_read = True

        decode_attention_task = ComputeTask(
            request_id = request.request_id,
            task_type = "attention",
            workload = {
                "q_length": 1,
                "kv_length": i + 3,
            },
        )
        decode_attention_result = chip.vex.execute(
            decode_attention_task,
            request_cycle = decode_read_result["finish_cycle"],
        )
        assert (
            decode_attention_result["request_cycle"]
            == decode_read_result["finish_cycle"]
        )
        assert decode_attention_result["compute_workload"]["q_length"] == 1
        assert decode_attention_result["compute_workload"]["kv_length"] == i + 3

        new_token_block = KVcacheBlock(
            block_id = f"synthetic-request-0-decode-kv-{i + 3}",
            request_id = request.request_id,
            layer_id = 0,
            first_token_position = i + 3,
            num_tokens = 1,
            size_byte = kv_size_per_token,
            chip_id = chip.chip_id,
        )
        new_token_block_store_result = chip.kv_cache_manager.store_kv_block(
            new_token_block,
            request_cycle = decode_attention_result["finish_cycle"],
        )
        assert new_token_block_store_result is not False
        assert (
            new_token_block_store_result["request_cycle"]
            == decode_attention_result["finish_cycle"]
        )

        if new_token_block.storage_location == "hbm":
            if first_hbm_token_position is None:
                first_hbm_token_position = new_token_block.first_token_position
        else:
            assert new_token_block.storage_location == "attention_buffer"

        previous_store_result = new_token_block_store_result
        request.generated_token_num += 1
        request.current_token_position += 1
        kv_cache_ids.append(new_token_block.block_id)

    # The final generated token is not decoded, so it does not need a KV block.
    request.status = "finished"
    request.finish_cycle = previous_store_result["finish_cycle"]

    stored_blocks = chip.kv_cache_manager.kv_cache_blocks.values()
    attention_buffer_stored_size = sum(
        block.size_byte
        for block in stored_blocks
        if block.storage_location == "attention_buffer"
    )
    hbm_stored_size = sum(
        block.size_byte
        for block in stored_blocks
        if block.storage_location == "hbm"
    )
    total_stored_token_num = (
        request.input_token_num + request.output_token_num - 1
    )

    assert saw_attention_buffer_only_read
    assert saw_mixed_read
    assert first_hbm_token_position == 32
    assert chip.attention_buffer.usage_byte == attention_buffer_stored_size
    assert chip.hbm.usage_byte == hbm_stored_size
    assert chip.attention_buffer.usage_byte == 63_488
    assert chip.hbm.usage_byte == 40_960
    assert attention_buffer_stored_size + hbm_stored_size == (
        total_stored_token_num * kv_size_per_token
    )
    assert chip.attention_buffer.usage_byte <= chip.attention_buffer.size_byte
    assert chip.hbm.usage_byte > 0
    assert chip.kv_cache_manager.check_consistency()

    assert request.generated_token_num == request.output_token_num
    assert request.current_token_position == 52
    assert request.phase == "decode"
    assert request.status == "finished"
    assert request.start_cycle == request.arrival_cycle
    assert request.finish_cycle == previous_store_result["finish_cycle"]

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
        f"start={first_decode_start_cycle}, finish={request.finish_cycle}"
    )
    print(
        "KV overflow: "
        f"first_hbm_token_position={first_hbm_token_position}, "
        f"mixed_read={saw_mixed_read}"
    )
    print(
        "Memory usage: "
        f"attention_buffer={chip.attention_buffer.usage_byte}/"
        f"{chip.attention_buffer.size_byte} bytes, "
        f"hbm={chip.hbm.usage_byte} bytes"
    )
    print(
        "Final Request state: "
        f"generated_token_num={request.generated_token_num}, "
        f"current_token_position={request.current_token_position}, "
        f"phase={request.phase}, status={request.status}, "
        f"finish_cycle={request.finish_cycle}"
    )


if __name__ == "__main__":
    test_request_chip_long_context_hbm_overflow()
