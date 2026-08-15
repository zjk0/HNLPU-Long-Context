import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from chip import Chip  # noqa: E402
from compute import ComputeTask  # noqa: E402
from config import Config  # noqa: E402
from kv_cache import KVcacheBlock  # noqa: E402
from request import Request  # noqa: E402


def test_request_chip_multilayer_kv_isolation():
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
    layer_ids = [0, 1, 2]

    kv_size_per_token = (
        2
        * config.model["num_kv_heads"]
        * config.model["head_dim"]
        * config.memory["kv_dtype_bytes"]
    )

    request.status = "running"
    request.phase = "prefill"
    request.start_cycle = request.arrival_cycle

    current_cycle = request.arrival_cycle
    prefill_results = {}
    prefill_block_ids = {}

    # Run the two-token prefill serially through the first three layers.
    for layer_id in layer_ids:
        projection_task = ComputeTask(
            request_id = request.request_id,
            task_type = "projection",
            workload = {
                "projection_type": "qkv",
                "num_tokens": request.input_token_num,
            },
        )
        projection_result = chip.hn_array.execute(
            projection_task,
            request_cycle = current_cycle,
        )
        assert projection_result["request_cycle"] == current_cycle

        attention_task = ComputeTask(
            request_id = request.request_id,
            task_type = "attention",
            workload = {
                "q_length": 2,
                "kv_length": 2,
            },
        )
        attention_result = chip.vex.execute(
            attention_task,
            request_cycle = projection_result["finish_cycle"],
        )
        assert (
            attention_result["request_cycle"]
            == projection_result["finish_cycle"]
        )

        block_id = (
            f"synthetic-request-0-layer-{layer_id}-prefill-kv"
        )
        prefill_block = KVcacheBlock(
            block_id = block_id,
            request_id = request.request_id,
            layer_id = layer_id,
            first_token_position = 1,
            num_tokens = 2,
            size_byte = 2 * kv_size_per_token,
            chip_id = chip.chip_id,
        )
        store_result = chip.kv_cache_manager.store_kv_block(
            prefill_block,
            request_cycle = attention_result["finish_cycle"],
        )
        assert store_result is not False
        assert store_result["request_cycle"] == attention_result["finish_cycle"]
        assert prefill_block.storage_location == "attention_buffer"

        request_layer_key = (request.request_id, layer_id)
        assert chip.kv_cache_manager.request_layer_blocks[request_layer_key] == [
            block_id
        ]

        prefill_block_ids[layer_id] = block_id
        prefill_results[layer_id] = {
            "projection": projection_result,
            "attention": attention_result,
            "store": store_result,
        }
        current_cycle = store_result["finish_cycle"]

    expected_layer_keys = {
        (request.request_id, layer_id) for layer_id in layer_ids
    }
    assert expected_layer_keys.issubset(
        chip.kv_cache_manager.request_layer_blocks
    )
    assert len(set(prefill_block_ids.values())) == len(layer_ids)

    for layer_id in layer_ids:
        layer_block_ids = chip.kv_cache_manager.request_layer_blocks[
            (request.request_id, layer_id)
        ]
        assert layer_block_ids == [prefill_block_ids[layer_id]]
        assert all(
            chip.kv_cache_manager.kv_cache_blocks[block_id].layer_id
            == layer_id
            for block_id in layer_block_ids
        )

    assert chip.attention_buffer.usage_byte == (
        len(layer_ids) * 2 * kv_size_per_token
    )
    assert chip.hbm.usage_byte == 0

    # Prefill produces token 3, which is decoded through the same three layers.
    request.generated_token_num = 1
    request.current_token_position = 3
    request.phase = "decode"

    decode_results = {}
    decode_block_ids = {}

    for layer_id in layer_ids:
        projection_task = ComputeTask(
            request_id = request.request_id,
            task_type = "projection",
            workload = {
                "projection_type": "qkv",
                "num_tokens": 1,
            },
        )
        projection_result = chip.hn_array.execute(
            projection_task,
            request_cycle = current_cycle,
        )
        assert projection_result["request_cycle"] == current_cycle

        request_layer_key = (request.request_id, layer_id)
        historical_block_ids = list(
            chip.kv_cache_manager.request_layer_blocks[request_layer_key]
        )
        assert historical_block_ids == [prefill_block_ids[layer_id]]
        assert all(
            chip.kv_cache_manager.kv_cache_blocks[block_id].layer_id
            == layer_id
            for block_id in historical_block_ids
        )

        read_result = chip.kv_cache_manager.read_kv_blocks(
            historical_block_ids,
            request_cycle = projection_result["finish_cycle"],
        )
        assert read_result["request_cycle"] == projection_result["finish_cycle"]
        assert read_result["total_read_size_byte"] == 2 * kv_size_per_token
        assert read_result["attention_buffer_result"] is not None
        assert read_result["hbm_result"] is None

        attention_task = ComputeTask(
            request_id = request.request_id,
            task_type = "attention",
            workload = {
                "q_length": 1,
                "kv_length": 3,
            },
        )
        attention_result = chip.vex.execute(
            attention_task,
            request_cycle = read_result["finish_cycle"],
        )
        assert attention_result["request_cycle"] == read_result["finish_cycle"]
        assert attention_result["compute_workload"]["q_length"] == 1
        assert attention_result["compute_workload"]["kv_length"] == 3

        block_id = (
            f"synthetic-request-0-layer-{layer_id}-token-3-kv"
        )
        token_3_block = KVcacheBlock(
            block_id = block_id,
            request_id = request.request_id,
            layer_id = layer_id,
            first_token_position = 3,
            num_tokens = 1,
            size_byte = kv_size_per_token,
            chip_id = chip.chip_id,
        )
        store_result = chip.kv_cache_manager.store_kv_block(
            token_3_block,
            request_cycle = attention_result["finish_cycle"],
        )
        assert store_result is not False
        assert store_result["request_cycle"] == attention_result["finish_cycle"]
        assert token_3_block.storage_location == "attention_buffer"

        decode_block_ids[layer_id] = block_id
        decode_results[layer_id] = {
            "projection": projection_result,
            "read": read_result,
            "attention": attention_result,
            "store": store_result,
        }
        current_cycle = store_result["finish_cycle"]

    request.generated_token_num = 2
    request.current_token_position = 4
    request.status = "finished"
    request.finish_cycle = current_cycle

    all_block_ids = set()
    total_kv_size_byte = 0
    blocks_per_layer = {}

    for layer_id in layer_ids:
        request_layer_key = (request.request_id, layer_id)
        layer_block_ids = chip.kv_cache_manager.request_layer_blocks[
            request_layer_key
        ]
        assert layer_block_ids == [
            prefill_block_ids[layer_id],
            decode_block_ids[layer_id],
        ]

        layer_blocks = [
            chip.kv_cache_manager.kv_cache_blocks[block_id]
            for block_id in layer_block_ids
        ]
        assert [block.first_token_position for block in layer_blocks] == [1, 3]
        assert all(block.layer_id == layer_id for block in layer_blocks)
        assert all(
            block.storage_location == "attention_buffer"
            for block in layer_blocks
        )

        layer_kv_size_byte = sum(block.size_byte for block in layer_blocks)
        assert layer_kv_size_byte == 3 * kv_size_per_token
        total_kv_size_byte += layer_kv_size_byte
        all_block_ids.update(layer_block_ids)
        blocks_per_layer[layer_id] = layer_block_ids

    assert len(all_block_ids) == 2 * len(layer_ids)
    assert chip.kv_cache_manager.request_blocks[request.request_id] == all_block_ids
    assert total_kv_size_byte == 3 * len(layer_ids) * kv_size_per_token
    assert chip.attention_buffer.usage_byte == total_kv_size_byte
    assert chip.hbm.usage_byte == 0
    assert chip.kv_cache_manager.check_consistency()

    assert request.generated_token_num == 2
    assert request.current_token_position == 4
    assert request.phase == "decode"
    assert request.status == "finished"
    assert request.start_cycle == request.arrival_cycle
    assert request.finish_cycle == decode_results[layer_ids[-1]]["store"][
        "finish_cycle"
    ]

    print("Prefill:")
    for layer_id in layer_ids:
        result = prefill_results[layer_id]
        print(
            f"layer{layer_id}: "
            f"projection={result['projection']['start_cycle']}-"
            f"{result['projection']['finish_cycle']}, "
            f"attention={result['attention']['start_cycle']}-"
            f"{result['attention']['finish_cycle']}, "
            f"store={result['store']['start_cycle']}-"
            f"{result['store']['finish_cycle']}"
        )

    print("\nDecode token3:")
    for layer_id in layer_ids:
        result = decode_results[layer_id]
        print(
            f"layer{layer_id}: "
            f"read={result['read']['start_cycle']}-"
            f"{result['read']['finish_cycle']} "
            f"({result['read']['total_read_size_byte']} bytes), "
            f"attention={result['attention']['start_cycle']}-"
            f"{result['attention']['finish_cycle']}, "
            f"store={result['store']['start_cycle']}-"
            f"{result['store']['finish_cycle']}"
        )

    print("\nKV blocks per layer:")
    for layer_id in layer_ids:
        print(f"layer{layer_id}: {blocks_per_layer[layer_id]}")

    print(f"\nAttention Buffer usage: {chip.attention_buffer.usage_byte} bytes")
    print(f"HBM usage: {chip.hbm.usage_byte} bytes")
    print(
        "Final Request state: "
        f"generated_token_num={request.generated_token_num}, "
        f"current_token_position={request.current_token_position}, "
        f"phase={request.phase}, status={request.status}, "
        f"finish_cycle={request.finish_cycle}"
    )


if __name__ == "__main__":
    test_request_chip_multilayer_kv_isolation()
