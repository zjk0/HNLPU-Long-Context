import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402


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
    attention_rmsnorm_task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": "rmsnorm"},
        layer_id = layer_id,
    )
    attention_rmsnorm_result = chip.vex.execute(
        attention_rmsnorm_task,
        request_cycle = request_cycle,
    )

    qkv_request_cycle = attention_rmsnorm_result["finish_cycle"]
    q_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "q",
    )
    k_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "k",
    )
    v_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "v",
    )
    q_result = chip.hn_array.execute(q_task, request_cycle = qkv_request_cycle)
    k_result = chip.hn_array.execute(k_task, request_cycle = qkv_request_cycle)
    v_result = chip.hn_array.execute(v_task, request_cycle = qkv_request_cycle)
    qkv_finish_cycle = max(
        q_result["finish_cycle"],
        k_result["finish_cycle"],
        v_result["finish_cycle"],
    )

    assert q_result["request_cycle"] == qkv_request_cycle
    assert k_result["request_cycle"] == qkv_request_cycle
    assert v_result["request_cycle"] == qkv_request_cycle
    assert q_result["weight_type"] == "q"
    assert k_result["weight_type"] == "k"
    assert v_result["weight_type"] == "v"

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

    xo_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "xo",
    )
    xo_result = chip.hn_array.execute(
        xo_task,
        request_cycle = attention_result["finish_cycle"],
    )
    assert xo_result["request_cycle"] == attention_result["finish_cycle"]

    attention_residual_task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": "residual"},
        layer_id = layer_id,
    )
    attention_residual_result = chip.vex.execute(
        attention_residual_task,
        request_cycle = xo_result["finish_cycle"],
    )
    assert attention_residual_result["request_cycle"] == xo_result["finish_cycle"]

    moe_rmsnorm_task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": "rmsnorm"},
        layer_id = layer_id,
    )
    moe_rmsnorm_result = chip.vex.execute(
        moe_rmsnorm_task,
        request_cycle = attention_residual_result["finish_cycle"],
    )
    assert (
        moe_rmsnorm_result["request_cycle"]
        == attention_residual_result["finish_cycle"]
    )

    router_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "router",
    )
    router_result = chip.hn_array.execute(
        router_task,
        request_cycle = moe_rmsnorm_result["finish_cycle"],
    )
    assert router_result["request_cycle"] == moe_rmsnorm_result["finish_cycle"]

    # This deterministic local expert only exercises expert-specific HNArray
    # routing. It is not a real GPT-OSS router result.
    selected_expert_id = chip.hn_array.expert_ids[0]
    up_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "up",
        expert_id = selected_expert_id,
    )
    gate_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "gate",
        expert_id = selected_expert_id,
    )
    up_result = chip.hn_array.execute(
        up_task,
        request_cycle = router_result["finish_cycle"],
    )
    gate_result = chip.hn_array.execute(
        gate_task,
        request_cycle = router_result["finish_cycle"],
    )
    up_gate_finish_cycle = max(
        up_result["finish_cycle"],
        gate_result["finish_cycle"],
    )

    assert up_result["request_cycle"] == router_result["finish_cycle"]
    assert gate_result["request_cycle"] == router_result["finish_cycle"]
    assert up_result["expert_id"] == selected_expert_id
    assert gate_result["expert_id"] == selected_expert_id

    swiglu_task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": "swiglu"},
        layer_id = layer_id,
    )
    swiglu_result = chip.vex.execute(
        swiglu_task,
        request_cycle = up_gate_finish_cycle,
    )
    assert swiglu_result["request_cycle"] == up_gate_finish_cycle

    down_task = ComputeTask(
        request_id = request_id,
        task_type = "linear",
        workload = {"num_tokens": num_tokens},
        layer_id = layer_id,
        weight_type = "down",
        expert_id = selected_expert_id,
    )
    down_result = chip.hn_array.execute(
        down_task,
        request_cycle = swiglu_result["finish_cycle"],
    )
    assert down_result["request_cycle"] == swiglu_result["finish_cycle"]

    moe_residual_task = ComputeTask(
        request_id = request_id,
        task_type = "vector",
        workload = {"vector_op": "residual"},
        layer_id = layer_id,
    )
    moe_residual_result = chip.vex.execute(
        moe_residual_task,
        request_cycle = down_result["finish_cycle"],
    )
    assert moe_residual_result["request_cycle"] == down_result["finish_cycle"]

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

    # Prefill computes both input tokens without reading historical KV.
    prefill_result = _execute_transformer_layer(
        chip = chip,
        request_id = request.request_id,
        layer_id = 0,
        num_tokens = request.input_token_num,
        q_length = 2,
        kv_length = 2,
        request_cycle = request.arrival_cycle,
    )
    prefill_q_result = prefill_result["q"]
    prefill_k_result = prefill_result["k"]
    prefill_v_result = prefill_result["v"]
    prefill_attention_result = prefill_result["attention"]
    prefill_xo_result = prefill_result["xo"]
    prefill_router_result = prefill_result["router"]
    prefill_up_result = prefill_result["up"]
    prefill_gate_result = prefill_result["gate"]
    prefill_swiglu_result = prefill_result["swiglu"]
    prefill_down_result = prefill_result["down"]

    assert prefill_result["attention_rmsnorm"]["request_cycle"] == request.arrival_cycle
    assert prefill_q_result["request_cycle"] == prefill_k_result["request_cycle"]
    assert prefill_q_result["request_cycle"] == prefill_v_result["request_cycle"]
    assert prefill_attention_result["request_cycle"] == prefill_result["qkv_finish_cycle"]
    assert prefill_xo_result["request_cycle"] == prefill_attention_result["finish_cycle"]
    assert prefill_swiglu_result["request_cycle"] == max(
        prefill_up_result["finish_cycle"],
        prefill_gate_result["finish_cycle"],
    )
    assert prefill_down_result["request_cycle"] == prefill_swiglu_result["finish_cycle"]

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

    prefill_ab_usage_byte = chip.attention_buffer.usage_byte
    assert prefill_kv_block.storage_location == "attention_buffer"
    assert prefill_kv_block.allocate_id in chip.attention_buffer.allocate_info
    assert prefill_kv_block.allocate_id not in chip.hbm.allocate_info
    assert chip.hbm.usage_byte == 0
    assert prefill_ab_usage_byte == 2 * kv_size_per_token

    # Token 3 is the abstract output of prefill and becomes the decode input.
    request.generated_token_num = 1
    request.current_token_position = 3
    request.phase = "decode"

    decode_result = _execute_transformer_layer(
        chip = chip,
        request_id = request.request_id,
        layer_id = 0,
        num_tokens = 1,
        q_length = 1,
        kv_length = 3,
        request_cycle = prefill_store_result["finish_cycle"],
        historical_block_ids = [prefill_kv_block.block_id],
    )
    decode_read_result = decode_result["read"]
    decode_attention_result = decode_result["attention"]

    assert decode_read_result is not None
    assert decode_read_result["attention_buffer_result"] is not None
    assert decode_read_result["hbm_result"] is None
    assert decode_read_result["total_read_size_byte"] == 2 * kv_size_per_token
    assert decode_read_result["request_cycle"] == decode_result["qkv_finish_cycle"]
    assert decode_attention_result["request_cycle"] == decode_read_result["finish_cycle"]
    assert decode_attention_result["compute_workload"]["q_length"] == 1
    assert decode_attention_result["compute_workload"]["kv_length"] == 3
    assert "synthetic-request-0-token-3-kv" not in chip.kv_cache_manager.kv_cache_blocks

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
        request_cycle = decode_result["finish_cycle"],
    )
    assert decode_store_result is not False
    assert decode_store_result["request_cycle"] == decode_result["finish_cycle"]
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
        f"qkv={prefill_q_result['start_cycle']}-"
        f"{prefill_result['qkv_finish_cycle']}, "
        f"attention={prefill_attention_result['start_cycle']}-"
        f"{prefill_attention_result['finish_cycle']}, "
        f"router={prefill_router_result['start_cycle']}-"
        f"{prefill_router_result['finish_cycle']}, "
        f"layer_finish={prefill_result['finish_cycle']}, "
        f"kv_store_finish={prefill_store_result['finish_cycle']}"
    )
    print(
        "Decode cycles: "
        f"history_kv_read={decode_read_result['start_cycle']}-"
        f"{decode_read_result['finish_cycle']}, "
        f"attention={decode_attention_result['start_cycle']}-"
        f"{decode_attention_result['finish_cycle']}, "
        f"layer_finish={decode_result['finish_cycle']}, "
        f"kv_store_finish={decode_store_result['finish_cycle']}"
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
