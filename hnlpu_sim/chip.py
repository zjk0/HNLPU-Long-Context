from memory import AttentionBuffer, HBM
from kv_cache import KVcacheManager
from config import Config
from compute import VEX, HNArray

class Chip:
    def __init__(self, row, column, config):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config.")

        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError("row must be an integer.")
        if row < 0:
            raise ValueError("row must be greater than or equal to 0.")
        if row >= config.hnlpu["chip_grid_rows"]:
            raise ValueError("row must be less than config.hnlpu['chip_grid_rows'].")

        if not isinstance(column, int) or isinstance(column, bool):
            raise TypeError("column must be an integer.")
        if column < 0:
            raise ValueError("column must be greater than or equal to 0.")
        if column >= config.hnlpu["chip_grid_cols"]:
            raise ValueError("column must be less than config.hnlpu['chip_grid_cols'].")

        self.row = row
        self.column = column
        self.config = config
        self.chip_id = (row, column)

        self.attention_buffer = AttentionBuffer(
            num_banks = config.memory["attention_buffer_banks_per_chip"],
            bank_size_byte = config.attention_buffer_bank_size_byte,
            access_latency_cycles = config.memory["attention_buffer_latency_cycles"],
            clock_frequency_hz = config.clock_frequency_hz,
        )

        self.hbm = HBM(
            bandwidth_byte_per_s = config.hbm_bandwidth_byte_per_s,
            clock_frequency_hz = config.clock_frequency_hz,
        )
        if self.hbm.size_byte != config.hbm_size_byte:
            raise ValueError("Config HBM capacity does not match HBM's default stack topology.")

        self.kv_cache_manager = KVcacheManager(
            chip_id = self.chip_id,
            attention_buffer = self.attention_buffer,
            hbm = self.hbm,
        )
        self.vex = VEX(
            layer_num = config.model["num_layers"],
            cached_kv_heads_per_cycle = config.hnlpu["vex_cached_kv_heads_per_cycle_per_chip"],
            fixed_latency_cycles = config.vex,
        )

        chip_linear_id = row * config.hnlpu["chip_grid_cols"] + column
        experts_per_chip = config.model["num_experts"] // config.hnlpu["num_chips"]
        expert_start_id = chip_linear_id * experts_per_chip

        # Simulation assumption: experts are evenly assigned in linear chip order.
        expert_ids = list(range(expert_start_id, expert_start_id + experts_per_chip))
        self.hn_array = HNArray(
            layer_num = config.model["num_layers"],
            expert_ids = expert_ids,
            weight_type_latency = config.hnlpu["hn_array_latency_cycles"],
        )
