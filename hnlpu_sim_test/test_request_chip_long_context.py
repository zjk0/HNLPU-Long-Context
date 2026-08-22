import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402


def _execute_linear(
    chip,
    request_id,
    layer_id,
    weight_type,
    num_tokens,
    request_cycle,
    expert_id = None,
):
    task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = weight_type,
        expert_id = expert_id,
    )
    result = chip.hn_array.execute(task, request_cycle = request_cycle)
    assert result["request_cycle"] == request_cycle
    assert result["layer_id"] == layer_id
    assert result["weight_type"] == weight_type
    assert result["expert_id"] == expert_id
    return result


def _execute_vector(chip, request_id, layer_id, vector_op, request_cycle):
    task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": vector_op},
        layer_id = layer_id,
    )
    result = chip.vex.execute(task, request_cycle = request_cycle)
    assert result["request_cycle"] == request_cycle
    assert result["layer_id"] == layer_id
    return result


def _execute_transformer_layer(
    chip,
    request_id,
    layer_id,
    num_tokens,
    q_length,
    kv_length,
    request_cycle,
    historical_block_ids = None,
):
    attention_rmsnorm_result = _execute_vector(
        chip,
        request_id,
        layer_id,
        "rmsnorm",
        request_cycle,
    )

    qkv_request_cycle = attention_rmsnorm_result["finish_cycle"]
    q_result = _execute_linear(
        chip, request_id, layer_id, "q", num_tokens, qkv_request_cycle
    )
    k_result = _execute_linear(
        chip, request_id, layer_id, "k", num_tokens, qkv_request_cycle
    )
    v_result = _execute_linear(
        chip, request_id, layer_id, "v", num_tokens, qkv_request_cycle
    )
    qkv_finish_cycle = max(
        q_result["finish_cycle"],
        k_result["finish_cycle"],
        v_result["finish_cycle"],
    )

    read_result = None
    attention_request_cycle = qkv_finish_cycle
    if historical_block_ids is not None:
        read_result = chip.kv_cache_manager.read_kv_blocks(
            historical_block_ids,
            request_cycle = qkv_finish_cycle,
        )
        assert read_result["request_cycle"] == qkv_finish_cycle
        attention_request_cycle = read_result["finish_cycle"]

    attention_task = ComputeTask(
        request_id = request_id,
        task_type = "attention",
        workload = {
            "q_length": q_length,
            "kv_length": kv_length,
        },
        layer_id = layer_id,
    )
    attention_result = chip.vex.execute(
        attention_task,
        request_cycle = attention_request_cycle,
    )
    assert attention_result["request_cycle"] == attention_request_cycle
    assert attention_result["layer_id"] == layer_id

    xo_result = _execute_linear(
        chip,
        request_id,
        layer_id,
        "xo",
        num_tokens,
        attention_result["finish_cycle"],
    )
    attention_residual_result = _execute_vector(
        chip,
        request_id,
        layer_id,
        "residual",
        xo_result["finish_cycle"],
    )
    moe_rmsnorm_result = _execute_vector(
        chip,
        request_id,
        layer_id,
        "rmsnorm",
        attention_residual_result["finish_cycle"],
    )
    router_result = _execute_linear(
        chip,
        request_id,
        layer_id,
        "router",
        num_tokens,
        moe_rmsnorm_result["finish_cycle"],
    )

    # This deterministic local expert only exercises expert-specific HNArray
    # routing. It is not a real GPT-OSS router result.
    selected_expert_id = chip.hn_array.expert_ids[0]
    up_result = _execute_linear(
        chip,
        request_id,
        layer_id,
        "up",
        num_tokens,
        router_result["finish_cycle"],
        selected_expert_id,
    )
    gate_result = _execute_linear(
        chip,
        request_id,
        layer_id,
        "gate",
        num_tokens,
        router_result["finish_cycle"],
        selected_expert_id,
    )
    up_gate_finish_cycle = max(
        up_result["finish_cycle"],
        gate_result["finish_cycle"],
    )

    swiglu_result = _execute_vector(
        chip,
        request_id,
        layer_id,
        "swiglu",
        up_gate_finish_cycle,
    )
    down_result = _execute_linear(
        chip,
        request_id,
        layer_id,
        "down",
        num_tokens,
        swiglu_result["finish_cycle"],
        selected_expert_id,
    )
    moe_residual_result = _execute_vector(
        chip,
        request_id,
        layer_id,
        "residual",
        down_result["finish_cycle"],
    )

    return {
        "attention_rmsnorm": attention_rmsnorm_result,
        "q": q_result,
        "k": k_result,
        "v": v_result,
        "qkv_finish_cycle": qkv_finish_cycle,
        "read": read_result,
        "attention": attention_result,
        "xo": xo_result,
        "attention_residual": attention_residual_result,
        "moe_rmsnorm": moe_rmsnorm_result,
        "router": router_result,
        "up": up_result,
        "gate": gate_result,
        "swiglu": swiglu_result,
        "down": down_result,
        "moe_residual": moe_residual_result,
        "selected_expert_id": selected_expert_id,
        "finish_cycle": moe_residual_result["finish_cycle"],
    }


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

    # Prefill computes both input tokens and does not read historical KV.
    prefill_result = _execute_transformer_layer(
        chip = chip,
        request_id = request.request_id,
        layer_id = 0,
        num_tokens = request.input_token_num,
        q_length = 2,
        kv_length = 2,
        request_cycle = request.arrival_cycle,
    )
    assert prefill_result["attention_rmsnorm"]["request_cycle"] == request.arrival_cycle
    assert prefill_result["q"]["request_cycle"] == prefill_result["k"]["request_cycle"]
    assert prefill_result["q"]["request_cycle"] == prefill_result["v"]["request_cycle"]
    assert prefill_result["attention"]["request_cycle"] == prefill_result["qkv_finish_cycle"]
    assert prefill_result["swiglu"]["request_cycle"] == max(
        prefill_result["up"]["finish_cycle"],
        prefill_result["gate"]["finish_cycle"],
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
        request_cycle = prefill_result["finish_cycle"],
    )
    assert prefill_store_result is not False
    assert prefill_store_result["request_cycle"] == prefill_result["finish_cycle"]
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
    last_decode_result = None

    for i in range(request.output_token_num - 1):
        # Read only historical KV. Current K/V comes from this iteration's K/V units.
        decode_result = _execute_transformer_layer(
            chip = chip,
            request_id = request.request_id,
            layer_id = 0,
            num_tokens = 1,
            q_length = 1,
            kv_length = i + 3,
            request_cycle = previous_store_result["finish_cycle"],
            historical_block_ids = kv_cache_ids,
        )
        if first_decode_start_cycle is None:
            first_decode_start_cycle = decode_result["attention_rmsnorm"][
                "start_cycle"
            ]

        decode_read_result = decode_result["read"]
        decode_attention_result = decode_result["attention"]
        assert decode_read_result is not None
        assert (
            decode_result["attention_rmsnorm"]["request_cycle"]
            == previous_store_result["finish_cycle"]
        )
        assert decode_read_result["request_cycle"] == decode_result["qkv_finish_cycle"]
        assert decode_read_result["total_read_size_byte"] == (
            i + request.input_token_num
        ) * kv_size_per_token
        assert decode_attention_result["request_cycle"] == decode_read_result["finish_cycle"]
        assert decode_attention_result["compute_workload"]["q_length"] == 1
        assert decode_attention_result["compute_workload"]["kv_length"] == i + 3
        assert decode_result["swiglu"]["request_cycle"] == max(
            decode_result["up"]["finish_cycle"],
            decode_result["gate"]["finish_cycle"],
        )

        attention_buffer_result = decode_read_result["attention_buffer_result"]
        hbm_result = decode_read_result["hbm_result"]
        if attention_buffer_result is not None and hbm_result is None:
            saw_attention_buffer_only_read = True
        if attention_buffer_result is not None and hbm_result is not None:
            saw_mixed_read = True

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
            request_cycle = decode_result["finish_cycle"],
        )
        assert new_token_block_store_result is not False
        assert (
            new_token_block_store_result["request_cycle"]
            == decode_result["finish_cycle"]
        )

        if new_token_block.storage_location == "hbm":
            if first_hbm_token_position is None:
                first_hbm_token_position = new_token_block.first_token_position
        else:
            assert new_token_block.storage_location == "attention_buffer"

        previous_store_result = new_token_block_store_result
        last_decode_result = decode_result
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
    total_stored_token_num = request.input_token_num + request.output_token_num - 1

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
    assert last_decode_result is not None

    print(
        "Prefill cycles: "
        f"qkv_finish={prefill_result['qkv_finish_cycle']}, "
        f"attention={prefill_result['attention']['start_cycle']}-"
        f"{prefill_result['attention']['finish_cycle']}, "
        f"layer_finish={prefill_result['finish_cycle']}, "
        f"kv_store_finish={prefill_store_result['finish_cycle']}"
    )
    print(
        "Decode cycles: "
        f"start={first_decode_start_cycle}, finish={request.finish_cycle}, "
        f"last_attention={last_decode_result['attention']['start_cycle']}-"
        f"{last_decode_result['attention']['finish_cycle']}"
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
