import math

SUPPORTED_COMMUNICATION_OPERATIONS = (
    "broadcast",
    "reduce",
    "scatter",
    "gather",
    "all_reduce",
    "all_gather",
)

SUPPORTED_COMMUNICATION_DIRECTIONS = (
    "row",
    "column",
    "all",
)


class CommunicationTask:
    COMMUNICATION_OPERATIONS = SUPPORTED_COMMUNICATION_OPERATIONS
    COMMUNICATION_DIRECTIONS = SUPPORTED_COMMUNICATION_DIRECTIONS

    def __init__(
        self,
        request_id,
        operation,
        participants,
        data_size_byte,
        source_chip = None,
        destination_chip = None,
        direction = None,
    ):
        if request_id is None or (isinstance(request_id, str) and not request_id.strip()):
            raise ValueError("request_id must not be empty.")
        try:
            hash(request_id)
        except TypeError as exc:
            raise TypeError("request_id must be hashable.") from exc

        if not isinstance(operation, str):
            raise TypeError("operation must be a string.")
        if not operation.strip():
            raise ValueError("operation must not be empty.")
        if operation not in self.COMMUNICATION_OPERATIONS:
            raise ValueError(f"Unsupported communication operation({operation}).")

        if not isinstance(participants, list):
            raise TypeError("participants must be a list.")
        if not participants:
            raise ValueError("participants must not be empty.")
        for participant in participants:
            if not isinstance(participant, int) or isinstance(participant, bool):
                raise TypeError("Every participant must be an integer.")
            if participant < 0:
                raise ValueError("Every participant must be greater than or equal to 0.")
        if len(set(participants)) != len(participants):
            raise ValueError("participants must not contain duplicates.")

        if not isinstance(data_size_byte, int) or isinstance(data_size_byte, bool):
            raise TypeError("data_size_byte must be an integer.")
        if data_size_byte <= 0:
            raise ValueError("data_size_byte must be greater than 0.")

        for field_name, chip_id in (
            ("source_chip", source_chip),
            ("destination_chip", destination_chip),
        ):
            if chip_id is None:
                continue
            if not isinstance(chip_id, int) or isinstance(chip_id, bool):
                raise TypeError(f"{field_name} must be an integer or None.")
            if chip_id < 0:
                raise ValueError(f"{field_name} must be greater than or equal to 0.")
            if chip_id not in participants:
                raise ValueError(f"{field_name} must be present in participants.")

        if direction is not None:
            if not isinstance(direction, str):
                raise TypeError("direction must be a string or None.")
            if direction not in self.COMMUNICATION_DIRECTIONS:
                raise ValueError(f"Unsupported communication direction({direction}).")

        if operation in ("broadcast", "scatter"):
            if source_chip is None:
                raise ValueError(f"operation({operation}) requires source_chip.")
            if destination_chip is not None:
                raise ValueError(f"operation({operation}) requires destination_chip to be None.")
        elif operation in ("reduce", "gather"):
            if destination_chip is None:
                raise ValueError(f"operation({operation}) requires destination_chip.")
            if source_chip is not None:
                raise ValueError(f"operation({operation}) requires source_chip to be None.")
        else:
            if source_chip is not None or destination_chip is not None:
                raise ValueError(
                    f"operation({operation}) requires source_chip and "
                    "destination_chip to be None."
                )

        self.request_id = request_id
        self.operation = operation
        self.participants = participants.copy()
        self.data_size_byte = data_size_byte
        self.source_chip = source_chip
        self.destination_chip = destination_chip
        self.direction = direction


class Interconnect:
    COLLECTIVE_OPERATIONS = SUPPORTED_COMMUNICATION_OPERATIONS

    def __init__(
        self,
        chip_grid_rows,
        chip_grid_cols,
        link_bandwidth_GBps,
        link_latency_ns,
        clock_frequency_hz,
        collective_algorithms,
    ):
        self._validate_integer(chip_grid_rows, "chip_grid_rows", minimum = 1)
        self._validate_integer(chip_grid_cols, "chip_grid_cols", minimum = 1)
        self._validate_number(link_bandwidth_GBps, "link_bandwidth_GBps", minimum = 0, inclusive = False)
        self._validate_number(link_latency_ns, "link_latency_ns", minimum = 0)
        self._validate_number(clock_frequency_hz, "clock_frequency_hz", minimum = 0, inclusive = False)

        if not isinstance(collective_algorithms, dict):
            raise TypeError("collective_algorithms must be a dictionary.")

        missing_operations = [
            operation
            for operation in self.COLLECTIVE_OPERATIONS
            if operation not in collective_algorithms
        ]
        if missing_operations:
            raise KeyError(
                "collective_algorithms is missing operations: "
                + ", ".join(missing_operations)
            )

        for operation, algorithm in collective_algorithms.items():
            if not isinstance(algorithm, str):
                raise TypeError(f"collective_algorithms[{operation!r}] must be a string.")
            if not algorithm.strip():
                raise ValueError(f"collective_algorithms[{operation!r}] must not be empty.")

        self.chip_grid_rows = chip_grid_rows
        self.chip_grid_cols = chip_grid_cols
        self.num_chips = chip_grid_rows * chip_grid_cols
        self.link_bandwidth_GBps = link_bandwidth_GBps
        self.link_latency_ns = link_latency_ns
        self.clock_frequency_hz = clock_frequency_hz
        self.collective_algorithms = collective_algorithms.copy()

        self.link_bandwidth_byte_per_s = link_bandwidth_GBps * 1_000_000_000
        self.link_latency_cycles = math.ceil(link_latency_ns * clock_frequency_hz / 1_000_000_000)

        self.row_groups = self._build_row_groups()
        self.column_groups = self._build_column_groups()

        # Each canonical pair represents one physical link. The current
        # simulator abstraction makes both directions share the same busy
        # state; a future full-duplex model can use direction-specific states.
        self.link_busy_until_cycle = self._build_links()

    @staticmethod
    def _validate_integer(value, name, minimum = 0):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        if value < minimum:
            raise ValueError(f"{name} must be greater than or equal to {minimum}.")

    @staticmethod
    def _validate_number(value, name, minimum = 0, inclusive = True):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number.")
        if (inclusive and value < minimum) or (not inclusive and value <= minimum):
            comparison = "greater than or equal to" if inclusive else "greater than"
            raise ValueError(f"{name} must be {comparison} {minimum}.")

    def _get_chip_id(self, row, column):
        return row * self.chip_grid_cols + column

    def _build_row_groups(self):
        return {
            row: [self._get_chip_id(row, column) for column in range(self.chip_grid_cols)]
            for row in range(self.chip_grid_rows)
        }

    def _build_column_groups(self):
        return {
            column: [self._get_chip_id(row, column) for row in range(self.chip_grid_rows)]
            for column in range(self.chip_grid_cols)
        }

    def _build_links(self):
        link_busy_until_cycle = {}

        for groups in (self.row_groups, self.column_groups):
            for chip_ids in groups.values():
                for source_index, source_chip_id in enumerate(chip_ids):
                    for destination_chip_id in chip_ids[source_index + 1:]:
                        link_key = (
                            min(source_chip_id, destination_chip_id),
                            max(source_chip_id, destination_chip_id),
                        )
                        link_busy_until_cycle[link_key] = 0

        return link_busy_until_cycle
