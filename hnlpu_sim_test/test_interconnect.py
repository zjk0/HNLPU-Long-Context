import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HNLPU_SIM_PATH = PROJECT_ROOT / "hnlpu_sim"
sys.path.insert(0, str(HNLPU_SIM_PATH))

from config import Config  # noqa: E402
from interconnect import CommunicationTask, Interconnect  # noqa: E402


COLLECTIVE_ALGORITHMS = {
    "broadcast": "direct",
    "reduce": "direct",
    "scatter": "direct",
    "gather": "direct",
    "all_reduce": "direct",
    "all_gather": "direct",
}


def _create_interconnect(**overrides):
    parameters = {
        "chip_grid_rows": 4,
        "chip_grid_cols": 4,
        "link_bandwidth_GBps": 128,
        "link_latency_ns": 100,
        "clock_frequency_hz": 1_000_000_000,
        "collective_algorithms": COLLECTIVE_ALGORITHMS,
    }
    parameters.update(overrides)
    return Interconnect(**parameters)


def _create_communication_task(**overrides):
    parameters = {
        "request_id": "req-0",
        "operation": "all_reduce",
        "participants": [0, 1, 2, 3],
        "data_size_byte": 1024,
        "direction": "row",
    }
    parameters.update(overrides)
    return CommunicationTask(**parameters)


def test_initializes_reduce_communication_task_and_copies_participants():
    participants = [0, 4, 8, 12]
    task = CommunicationTask(
        request_id = "req-0",
        operation = "reduce",
        participants = participants,
        data_size_byte = 1024,
        destination_chip = 8,
        direction = "column",
    )

    assert task.request_id == "req-0"
    assert task.operation == "reduce"
    assert task.participants == [0, 4, 8, 12]
    assert task.data_size_byte == 1024
    assert task.source_chip is None
    assert task.destination_chip == 8
    assert task.direction == "column"

    participants.append(16)
    assert task.participants == [0, 4, 8, 12]


def test_initializes_all_reduce_communication_task_without_root_chips():
    task = _create_communication_task()

    assert task.operation == "all_reduce"
    assert task.source_chip is None
    assert task.destination_chip is None
    for excluded_field in (
        "algorithm",
        "request_cycle",
        "start_cycle",
        "finish_cycle",
        "wait_cycles",
        "service_cycles",
        "total_latency_cycles",
    ):
        assert not hasattr(task, excluded_field)


def test_initializes_broadcast_communication_task_with_source():
    task = CommunicationTask(
        request_id = "req-0",
        operation = "broadcast",
        participants = [0, 1, 2, 3],
        data_size_byte = 1024,
        source_chip = 2,
        direction = "row",
    )

    assert task.source_chip == 2
    assert task.destination_chip is None


@pytest.mark.parametrize(
    ("request_id", "expected_exception"),
    (
        (None, ValueError),
        ("", ValueError),
        ("  ", ValueError),
        ([], TypeError),
    ),
)
def test_communication_task_rejects_invalid_request_id(
    request_id,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(request_id = request_id)


@pytest.mark.parametrize(
    ("operation", "expected_exception"),
    (
        (None, TypeError),
        ("", ValueError),
        ("all_reduse", ValueError),
    ),
)
def test_communication_task_rejects_invalid_operation(
    operation,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(operation = operation)


@pytest.mark.parametrize(
    ("participants", "expected_exception"),
    (
        ((0, 1), TypeError),
        ([], ValueError),
        ([0, 1.0], TypeError),
        ([0, True], TypeError),
        ([0, -1], ValueError),
        ([0, 1, 1, 2], ValueError),
    ),
)
def test_communication_task_rejects_invalid_participants(
    participants,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(participants = participants)


@pytest.mark.parametrize(
    ("data_size_byte", "expected_exception"),
    (
        (1024.0, TypeError),
        (True, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ),
)
def test_communication_task_rejects_invalid_data_size(
    data_size_byte,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(data_size_byte = data_size_byte)


@pytest.mark.parametrize(
    ("source_chip", "expected_exception"),
    (
        ("0", TypeError),
        (True, TypeError),
        (-1, ValueError),
        (8, ValueError),
    ),
)
def test_communication_task_rejects_invalid_source_chip(
    source_chip,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(
            operation = "broadcast",
            source_chip = source_chip,
        )


@pytest.mark.parametrize(
    ("destination_chip", "expected_exception"),
    (
        ("0", TypeError),
        (False, TypeError),
        (-1, ValueError),
        (8, ValueError),
    ),
)
def test_communication_task_rejects_invalid_destination_chip(
    destination_chip,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(
            operation = "reduce",
            destination_chip = destination_chip,
        )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    (
        ({"operation": "broadcast"}, "requires source_chip"),
        ({"operation": "scatter"}, "requires source_chip"),
        ({"operation": "reduce"}, "requires destination_chip"),
        ({"operation": "gather"}, "requires destination_chip"),
        (
            {
                "operation": "broadcast",
                "source_chip": 0,
                "destination_chip": 1,
            },
            "requires destination_chip to be None",
        ),
        (
            {
                "operation": "scatter",
                "source_chip": 0,
                "destination_chip": 1,
            },
            "requires destination_chip to be None",
        ),
        (
            {
                "operation": "reduce",
                "source_chip": 0,
                "destination_chip": 1,
            },
            "requires source_chip to be None",
        ),
        (
            {
                "operation": "gather",
                "source_chip": 0,
                "destination_chip": 1,
            },
            "requires source_chip to be None",
        ),
        (
            {"operation": "all_reduce", "destination_chip": 1},
            "requires source_chip and destination_chip to be None",
        ),
        (
            {"operation": "all_gather", "source_chip": 0},
            "requires source_chip and destination_chip to be None",
        ),
    ),
)
def test_communication_task_enforces_operation_specific_root_semantics(
    overrides,
    expected_message,
):
    with pytest.raises(ValueError, match = expected_message):
        _create_communication_task(**overrides)


@pytest.mark.parametrize(
    ("direction", "expected_exception"),
    (
        (1, TypeError),
        ("diagonal", ValueError),
    ),
)
def test_communication_task_rejects_invalid_direction(
    direction,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_communication_task(direction = direction)


def test_communication_task_leaves_topology_consistency_to_interconnect():
    task = _create_communication_task(
        participants = [0, 5, 99],
        direction = "row",
    )

    assert task.participants == [0, 5, 99]


def test_initializes_4_by_4_row_column_fully_connected_topology():
    interconnect = _create_interconnect()

    assert interconnect.num_chips == 16
    assert interconnect.row_groups[0] == [0, 1, 2, 3]
    assert interconnect.row_groups[3] == [12, 13, 14, 15]
    assert interconnect.column_groups[0] == [0, 4, 8, 12]
    assert interconnect.column_groups[3] == [3, 7, 11, 15]

    for link_key in ((0, 1), (0, 3), (0, 4), (0, 12)):
        assert link_key in interconnect.link_busy_until_cycle
    for link_key in ((0, 5), (0, 10), (3, 12)):
        assert link_key not in interconnect.link_busy_until_cycle

    assert len(interconnect.link_busy_until_cycle) == 48
    assert all(
        source_chip_id < destination_chip_id
        for source_chip_id, destination_chip_id
        in interconnect.link_busy_until_cycle
    )
    assert all(
        busy_until_cycle == 0
        for busy_until_cycle in interconnect.link_busy_until_cycle.values()
    )


def test_converts_link_parameters_and_copies_collective_algorithms():
    collective_algorithms = COLLECTIVE_ALGORITHMS.copy()
    interconnect = _create_interconnect(
        collective_algorithms = collective_algorithms,
    )

    assert interconnect.link_bandwidth_GBps == 128
    assert interconnect.link_bandwidth_byte_per_s == 128_000_000_000
    assert interconnect.link_latency_ns == 100
    assert interconnect.link_latency_cycles == 100
    assert interconnect.clock_frequency_hz == 1_000_000_000
    assert interconnect.collective_algorithms == COLLECTIVE_ALGORITHMS

    collective_algorithms["broadcast"] = "tree"
    assert interconnect.collective_algorithms["broadcast"] == "direct"


def test_allows_zero_link_latency():
    interconnect = _create_interconnect(link_latency_ns = 0)

    assert interconnect.link_latency_cycles == 0


def test_repository_config_initializes_interconnect():
    config = Config(PROJECT_ROOT / "hnlpu_config.yaml")

    interconnect = Interconnect(
        chip_grid_rows = config.hnlpu["chip_grid_rows"],
        chip_grid_cols = config.hnlpu["chip_grid_cols"],
        link_bandwidth_GBps = config.interconnect[
            "cxl_bandwidth_GBps_per_link"
        ],
        link_latency_ns = config.interconnect["cxl_latency_ns"],
        clock_frequency_hz = config.clock_frequency_hz,
        collective_algorithms = config.interconnect["collective_algorithms"],
    )

    assert interconnect.num_chips == config.hnlpu["num_chips"]
    assert interconnect.link_bandwidth_byte_per_s == 128_000_000_000
    assert interconnect.link_latency_cycles == 100
    assert interconnect.collective_algorithms == COLLECTIVE_ALGORITHMS


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value", "expected_exception"),
    (
        ("chip_grid_rows", 0, ValueError),
        ("chip_grid_rows", 1.0, TypeError),
        ("chip_grid_cols", True, TypeError),
        ("link_bandwidth_GBps", 0, ValueError),
        ("link_bandwidth_GBps", -1, ValueError),
        ("link_bandwidth_GBps", "128", TypeError),
        ("link_latency_ns", -1, ValueError),
        ("link_latency_ns", False, TypeError),
        ("clock_frequency_hz", 0, ValueError),
        ("clock_frequency_hz", True, TypeError),
    ),
)
def test_rejects_invalid_scalar_parameters(
    parameter_name,
    invalid_value,
    expected_exception,
):
    with pytest.raises(expected_exception):
        _create_interconnect(**{parameter_name: invalid_value})


def test_rejects_non_dictionary_collective_algorithms():
    with pytest.raises(TypeError):
        _create_interconnect(collective_algorithms = [])


def test_rejects_missing_collective_algorithm():
    collective_algorithms = COLLECTIVE_ALGORITHMS.copy()
    collective_algorithms.pop("all_gather")

    with pytest.raises(KeyError, match = "all_gather"):
        _create_interconnect(
            collective_algorithms = collective_algorithms,
        )


def test_rejects_non_string_collective_algorithm():
    collective_algorithms = COLLECTIVE_ALGORITHMS.copy()
    collective_algorithms["reduce"] = None

    with pytest.raises(TypeError, match = "reduce"):
        _create_interconnect(
            collective_algorithms = collective_algorithms,
        )


def test_rejects_empty_collective_algorithm():
    collective_algorithms = COLLECTIVE_ALGORITHMS.copy()
    collective_algorithms["gather"] = "  "

    with pytest.raises(ValueError, match = "gather"):
        _create_interconnect(
            collective_algorithms = collective_algorithms,
        )
