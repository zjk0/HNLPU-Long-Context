import math

class Interconnect:
    COLLECTIVE_OPERATIONS = (
        "broadcast",
        "reduce",
        "scatter",
        "gather",
        "all_reduce",
        "all_gather",
    )

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
