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

    @staticmethod
    def _get_link_key(source_chip, destination_chip):
        return (min(source_chip, destination_chip), max(source_chip, destination_chip))

    def _calculate_link_transfer_cycles(self, data_size_byte):
        bandwidth_byte_per_cycle = self.link_bandwidth_byte_per_s / self.clock_frequency_hz
        serialization_cycles = max(1, math.ceil(data_size_byte / bandwidth_byte_per_cycle))

        # Simulator abstraction: a physical link remains busy throughout both
        # its fixed latency and serialization latency; packet pipelining is not
        # modeled yet.
        return self.link_latency_cycles + serialization_cycles

    def reduce(self, task: CommunicationTask, request_cycle):
        if not isinstance(task, CommunicationTask):
            raise TypeError("task must be a CommunicationTask.")
        if task.operation != "reduce":
            raise ValueError("Interconnect.reduce requires operation(reduce).")

        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")

        if not isinstance(task.participants, list):
            raise TypeError("task participants must be a list.")
        if not task.participants:
            raise ValueError("task participants must not be empty.")
        for participant in task.participants:
            if not isinstance(participant, int) or isinstance(participant, bool):
                raise TypeError("Every task participant must be an integer.")
            if not 0 <= participant < self.num_chips:
                raise ValueError("Every task participant must be a valid Interconnect chip ID.")
        if len(set(task.participants)) != len(task.participants):
            raise ValueError("task participants must not contain duplicates.")

        destination_chip = task.destination_chip
        if destination_chip is None:
            raise ValueError("A reduce task must provide destination_chip.")
        if not isinstance(destination_chip, int) or isinstance(destination_chip, bool):
            raise TypeError("task destination_chip must be an integer.")
        if not 0 <= destination_chip < self.num_chips:
            raise ValueError("task destination_chip must be a valid Interconnect chip ID.")
        if destination_chip not in task.participants:
            raise ValueError("task destination_chip must be present in participants.")
        if task.source_chip is not None:
            raise ValueError("A reduce task requires source_chip to be None.")

        if not isinstance(task.data_size_byte, int) or isinstance(task.data_size_byte, bool):
            raise TypeError("task data_size_byte must be an integer.")
        if task.data_size_byte <= 0:
            raise ValueError("task data_size_byte must be greater than 0.")

        participant_set = set(task.participants)
        if task.direction == "row":
            participant_rows = {
                chip_id // self.chip_grid_cols
                for chip_id in task.participants
            }
            if len(participant_rows) != 1:
                raise ValueError(
                    "reduce participants must belong to the same row when "
                    "direction is 'row'."
                )
        elif task.direction == "column":
            participant_columns = {
                chip_id % self.chip_grid_cols
                for chip_id in task.participants
            }
            if len(participant_columns) != 1:
                raise ValueError(
                    "reduce participants must belong to the same column when "
                    "direction is 'column'."
                )
        elif task.direction == "all":
            if participant_set != set(range(self.num_chips)):
                raise ValueError(
                    "reduce participants must contain every Interconnect chip "
                    "when direction is 'all'."
                )
        elif task.direction is not None:
            raise ValueError(f"Unsupported communication direction({task.direction}).")

        algorithm = self.collective_algorithms["reduce"]
        if algorithm != "direct":
            raise NotImplementedError(f"reduce algorithm({algorithm}) is not implemented.")

        required_links = []
        for source_chip in task.participants:
            if source_chip == destination_chip:
                continue

            # Canonical undirected keys implement the current simulator
            # abstraction in which both directions share one physical link
            # busy state.
            link_key = self._get_link_key(source_chip, destination_chip)
            if link_key not in self.link_busy_until_cycle:
                # Direct mode deliberately does not infer a multi-hop route.
                raise NotImplementedError(
                    "direct Reduce requires a physical point-to-point link "
                    "from every non-destination participant to the destination; "
                    f"no direct link exists from chip {source_chip} to chip "
                    f"{destination_chip}."
                )
            required_links.append(link_key)

        if not required_links:
            phase_start_cycle = request_cycle
            phase_finish_cycle = request_cycle
        else:
            # Simulator assumptions, not paper-specified timing facts: distinct
            # physical links transfer in parallel, and one phase waits until
            # every required link is available before all transfers start.
            phase_start_cycle = max(
                request_cycle,
                max(
                    self.link_busy_until_cycle[link]
                    for link in required_links
                ),
            )

            # data_size_byte is one complete partial result sent by each
            # non-destination participant, rather than their aggregate size.
            transfer_cycles = self._calculate_link_transfer_cycles(task.data_size_byte)
            phase_finish_cycle = phase_start_cycle + transfer_cycles

            # Simulator assumption: reduction arithmetic inside the
            # Interconnect Engine is negligible or fully overlapped with the
            # communication phase; no separate arithmetic latency is added.
            for link in required_links:
                self.link_busy_until_cycle[link] = phase_finish_cycle

        return {
            "request_id": task.request_id,
            "operation": "reduce",
            "algorithm": algorithm,
            "participants": task.participants.copy(),
            "destination_chip": destination_chip,
            "direction": task.direction,
            "data_size_byte": task.data_size_byte,
            "request_cycle": request_cycle,
            "start_cycle": phase_start_cycle,
            "finish_cycle": phase_finish_cycle,
            "wait_cycles": phase_start_cycle - request_cycle,
            "service_cycles": phase_finish_cycle - phase_start_cycle,
            "total_latency_cycles": phase_finish_cycle - request_cycle,
            "used_links": required_links.copy(),
        }
