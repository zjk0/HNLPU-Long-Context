import copy
import sys
from pathlib import Path

HNLPU_SIM_PATH = Path(__file__).resolve().parent.parent / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from memory import AttentionBuffer  # noqa: E402


def _snapshot_allocation_state(attention_buffer):
    return {
        "usage_byte": attention_buffer.usage_byte,
        "bank_group_usage_byte": list(
            attention_buffer.bank_group_usage_byte
        ),
        "bank_usage_byte": attention_buffer.bank_usage_byte.tolist(),
        "next_bank_group_id": attention_buffer.next_bank_group_id,
        "next_bank_group_offset": list(
            attention_buffer.next_bank_group_offset
        ),
        "allocate_info": copy.deepcopy(attention_buffer.allocate_info),
    }


def test_stage_1_balanced_single_group_allocation_still_has_priority():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 16,
        banks_per_group = 2,
        check_consistency = True,
    )

    assert attention_buffer.allocate_memory(8, "balanced")

    assert attention_buffer.allocate_info["balanced"] == {
        "size": 8,
        "bank_groups": {0: 8},
        "bank": {0: 4, 1: 4},
    }
    assert attention_buffer.next_bank_group_id == 1
    assert attention_buffer.next_bank_group_offset[0] == 0
    assert attention_buffer.check_consistency()


def test_stage_2_uses_round_robin_for_uneven_bank_capacity():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 7,
        banks_per_group = 4,
        check_consistency = True,
    )

    seed_ids = [f"seed-{index}" for index in range(4)]
    for seed_id in seed_ids:
        assert attention_buffer.allocate_memory(4, seed_id)
    for seed_id in seed_ids[1:]:
        assert attention_buffer.free_memory(seed_id)

    assert attention_buffer.bank_usage_byte.tolist() == [4, 0, 0, 0]
    assert attention_buffer.get_remaining_size() == 24

    assert attention_buffer.allocate_memory(12, "stage-2")
    allocation = attention_buffer.allocate_info["stage-2"]

    assert allocation["bank_groups"] == {0: 12}
    assert allocation["bank"] == {0: 3, 1: 4, 2: 4, 3: 1}
    assert allocation["bank"] != {0: 3, 1: 7, 2: 2}
    assert attention_buffer.next_bank_group_offset[0] == 0
    assert attention_buffer.check_consistency()


def test_stage_2_circular_traversal_accumulates_multiple_rounds():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 8,
        banks_per_group = 4,
        check_consistency = True,
    )

    seed_ids = [f"seed-{index}" for index in range(5)]
    for seed_id in seed_ids:
        assert attention_buffer.allocate_memory(4, seed_id)
    for seed_id in seed_ids[1:4]:
        assert attention_buffer.free_memory(seed_id)

    assert attention_buffer.bank_usage_byte.tolist() == [8, 0, 0, 0]
    assert attention_buffer.next_bank_group_offset[0] == 1

    assert attention_buffer.allocate_memory(16, "stage-2-multi-round")
    allocation = attention_buffer.allocate_info["stage-2-multi-round"]

    assert allocation["bank_groups"] == {0: 16}
    assert allocation["bank"] == {1: 8, 2: 4, 3: 4}
    assert allocation["bank"][1] == 2 * attention_buffer.access_width_byte
    assert attention_buffer.next_bank_group_offset[0] == 2
    assert attention_buffer.check_consistency()


def test_stage_3_spans_groups_when_no_single_group_is_large_enough():
    attention_buffer = AttentionBuffer(
        num_banks = 6,
        bank_size_byte = 8,
        banks_per_group = 3,
        check_consistency = True,
    )

    assert attention_buffer.allocate_memory(12, "cursor-seed-0")
    assert attention_buffer.free_memory("cursor-seed-0")
    assert attention_buffer.allocate_memory(12, "seed-1")
    assert attention_buffer.allocate_memory(12, "cursor-seed-1")
    assert attention_buffer.free_memory("cursor-seed-1")

    assert attention_buffer.bank_group_usage_byte == [0, 12]
    assert attention_buffer.get_remaining_size() == 36
    assert attention_buffer.next_bank_group_id == 1
    assert all(
        attention_buffer.bank_group_size_byte - usage < 25
        for usage in attention_buffer.bank_group_usage_byte
    )

    assert attention_buffer.allocate_memory(25, "stage-3")
    allocation = attention_buffer.allocate_info["stage-3"]

    assert allocation["bank_groups"] == {1: 12, 0: 13}
    assert allocation["bank"] == {
        3: 4,
        4: 4,
        5: 4,
        0: 5,
        1: 4,
        2: 4,
    }
    assert len(allocation["bank_groups"]) > 1
    assert {
        bank_id: allocation["bank"][bank_id]
        for bank_id in range(3)
    } == {0: 5, 1: 4, 2: 4}
    assert attention_buffer.next_bank_group_offset == [1, 0]
    assert attention_buffer.next_bank_group_id == 1
    assert attention_buffer.check_consistency()


def test_capacity_shortage_returns_false_without_changing_state():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 8,
        banks_per_group = 2,
        check_consistency = True,
    )
    assert attention_buffer.allocate_memory(28, "existing")
    assert attention_buffer.get_remaining_size() == 4
    state_before = _snapshot_allocation_state(attention_buffer)

    assert attention_buffer.allocate_memory(5, "overflow") is False

    assert _snapshot_allocation_state(attention_buffer) == state_before
    assert attention_buffer.check_consistency()


def test_free_restores_multi_group_allocation_capacity():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 8,
        banks_per_group = 2,
        check_consistency = True,
    )
    assert attention_buffer.allocate_memory(20, "multi-group")
    assert len(
        attention_buffer.allocate_info["multi-group"]["bank_groups"]
    ) > 1

    assert attention_buffer.free_memory("multi-group")

    assert attention_buffer.usage_byte == 0
    assert attention_buffer.bank_group_usage_byte == [0, 0]
    assert attention_buffer.bank_usage_byte.tolist() == [0, 0, 0, 0]
    assert attention_buffer.allocate_info == {}
    assert attention_buffer.check_consistency()


def test_read_and_write_cover_all_banks_in_multi_group_allocation():
    attention_buffer = AttentionBuffer(
        num_banks = 4,
        bank_size_byte = 8,
        banks_per_group = 2,
        check_consistency = True,
    )
    assert attention_buffer.allocate_memory(20, "multi-group")
    allocation = attention_buffer.allocate_info["multi-group"]
    involved_group_ids = {
        bank_id // attention_buffer.banks_per_group
        for bank_id in allocation["bank"]
    }
    assert len(involved_group_ids) > 1

    write_result = attention_buffer.write(
        "multi-group",
        request_cycle = 0,
    )
    read_result = attention_buffer.read(
        ["multi-group"],
        request_cycle = 0,
    )

    assert write_result["total_write_size_byte"] == 20
    assert read_result["total_read_size_byte"] == 20
    assert set(write_result["bank_write_size"]) == set(allocation["bank"])
    assert set(read_result["bank_read_size"]) == set(allocation["bank"])
    assert {
        bank_id // attention_buffer.banks_per_group
        for bank_id in write_result["bank_write_size"]
    } == involved_group_ids
    assert {
        bank_id // attention_buffer.banks_per_group
        for bank_id in read_result["bank_read_size"]
    } == involved_group_ids
    assert attention_buffer.check_consistency()
