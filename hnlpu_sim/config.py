from os import PathLike
from pathlib import Path
import yaml

class Config:
    def __init__(self, yaml_path, overrides = None):
        if not isinstance(yaml_path, (str, PathLike)):
            raise TypeError("yaml_path must be a path-like object.")
        if isinstance(yaml_path, str) and not yaml_path.strip():
            raise ValueError("yaml_path cannot be empty.")

        config_path = Path(yaml_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")

        try:
            with config_path.open("r", encoding = "utf-8") as config_file:
                config_data = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {config_path}") from exc

        if not isinstance(config_data, dict):
            raise ValueError("The YAML root must be a mapping.")

        if overrides is not None:
            if not isinstance(overrides, dict):
                raise TypeError("overrides must be a dictionary or None.")

            for section_name, section_overrides in overrides.items():
                if section_name not in config_data:
                    raise KeyError(
                        f"Override section does not exist in the YAML: {section_name}"
                    )
                if not isinstance(section_overrides, dict):
                    raise TypeError(
                        f"Override section '{section_name}' must be a dictionary."
                    )
                if not isinstance(config_data[section_name], dict):
                    raise TypeError(
                        f"Configuration section '{section_name}' must be a mapping."
                    )

                config_data[section_name].update(section_overrides)

        required_keys = {
            "model": (
                "name",
                "num_layers",
                "hidden_size",
                "num_q_heads",
                "num_kv_heads",
                "head_dim",
                "weight_bits",
                "num_weight_values",
                "num_experts",
                "top_k_experts",
                "vocab_size",
            ),
            "hnlpu": (
                "num_chips",
                "chip_grid_rows",
                "chip_grid_cols",
                "stages_per_layer",
                "pipeline_slots",
                "max_batch_size",
                "clock_GHz",
                "vex_cached_kv_heads_per_cycle_per_chip",
                "hn_array_latency_cycles",
                "reported_throughput_tokens_per_s_at_2k",
                "calibration_context_length",
                "calibration_batch_size",
            ),
            "vex": (
                "swiglu_latency_cycles",
                "rmsnorm_latency_cycles",
                "residual_latency_cycles",
                "sampling_latency_cycles",
            ),
            "memory": (
                "attention_buffer_mb_per_chip",
                "attention_buffer_banks_per_chip",
                "attention_buffer_bank_kb",
                "attention_buffer_bandwidth_TBps_per_chip",
                "attention_buffer_latency_cycles",
                "hbm_capacity_GB_per_chip",
                "hbm_bandwidth_GBps_per_chip",
                "kv_dtype_bytes",
            ),
            "interconnect": (
                "cxl_bandwidth_GBps_per_link",
                "cxl_latency_ns",
            ),
            "eval": (
                "context_lengths",
                "attention_time_fraction_at_calibration",
                "stall_hidden_until_context",
                "exposed_hbm_stall_fraction_at_512k",
                "output_csv",
                "output_plot",
                "memory_bandwidth_for_latency",
            ),
        }

        for section_name, section_keys in required_keys.items():
            if section_name not in config_data:
                raise KeyError(f"Missing configuration section: {section_name}")

            section = config_data[section_name]
            if not isinstance(section, dict):
                raise TypeError(f"Configuration section '{section_name}' must be a mapping.")

            missing_keys = [key for key in section_keys if key not in section]
            if missing_keys:
                missing_keys_text = ", ".join(missing_keys)
                raise KeyError(
                    f"Missing keys in configuration section '{section_name}': "
                    f"{missing_keys_text}"
                )

        context_lengths = config_data["eval"]["context_lengths"]
        if not isinstance(context_lengths, dict):
            raise TypeError("eval.context_lengths must be a mapping.")
        missing_context_keys = [
            key for key in ("start", "stop", "num") if key not in context_lengths
        ]
        if missing_context_keys:
            missing_keys_text = ", ".join(missing_context_keys)
            raise KeyError(f"Missing keys in eval.context_lengths: {missing_keys_text}")

        hn_array_latency_cycles = config_data["hnlpu"]["hn_array_latency_cycles"]
        if not isinstance(hn_array_latency_cycles, dict):
            raise TypeError("hnlpu.hn_array_latency_cycles must be a mapping.")
        hn_array_weight_types = (
            "q",
            "k",
            "v",
            "xo",
            "router",
            "up",
            "gate",
            "down",
        )
        missing_weight_types = [
            weight_type
            for weight_type in hn_array_weight_types
            if weight_type not in hn_array_latency_cycles
        ]
        if missing_weight_types:
            missing_weight_types_text = ", ".join(missing_weight_types)
            raise KeyError(
                "Missing keys in hnlpu.hn_array_latency_cycles: "
                f"{missing_weight_types_text}"
            )
        for weight_type in hn_array_weight_types:
            latency_cycles = hn_array_latency_cycles[weight_type]
            if not isinstance(latency_cycles, int) or isinstance(latency_cycles, bool):
                raise TypeError(
                    "hnlpu.hn_array_latency_cycles."
                    f"{weight_type} must be an integer."
                )
            if latency_cycles <= 0:
                raise ValueError(
                    "hnlpu.hn_array_latency_cycles."
                    f"{weight_type} must be greater than 0."
                )

        integer_fields = {
            "model": (
                "num_layers",
                "hidden_size",
                "num_q_heads",
                "num_kv_heads",
                "head_dim",
                "weight_bits",
                "num_weight_values",
                "num_experts",
                "top_k_experts",
                "vocab_size",
            ),
            "hnlpu": (
                "num_chips",
                "chip_grid_rows",
                "chip_grid_cols",
                "stages_per_layer",
                "pipeline_slots",
                "max_batch_size",
                "vex_cached_kv_heads_per_cycle_per_chip",
                "reported_throughput_tokens_per_s_at_2k",
                "calibration_context_length",
                "calibration_batch_size",
            ),
            "vex": (
                "swiglu_latency_cycles",
                "rmsnorm_latency_cycles",
                "residual_latency_cycles",
                "sampling_latency_cycles",
            ),
            "memory": (
                "attention_buffer_banks_per_chip",
                "attention_buffer_bank_kb",
                "attention_buffer_latency_cycles",
                "hbm_capacity_GB_per_chip",
                "hbm_bandwidth_GBps_per_chip",
                "kv_dtype_bytes",
            ),
        }
        for section_name, field_names in integer_fields.items():
            for field_name in field_names:
                value = config_data[section_name][field_name]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{section_name}.{field_name} must be an integer.")
                if value <= 0:
                    raise ValueError(f"{section_name}.{field_name} must be greater than 0.")

        positive_number_fields = {
            "hnlpu": ("clock_GHz",),
            "memory": (
                "attention_buffer_mb_per_chip",
                "attention_buffer_bandwidth_TBps_per_chip",
            ),
            "interconnect": (
                "cxl_bandwidth_GBps_per_link",
                "cxl_latency_ns",
            ),
        }
        for section_name, field_names in positive_number_fields.items():
            for field_name in field_names:
                value = config_data[section_name][field_name]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{section_name}.{field_name} must be a number.")
                if value <= 0:
                    raise ValueError(f"{section_name}.{field_name} must be greater than 0.")

        for field_name in ("start", "stop", "num"):
            value = context_lengths[field_name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"eval.context_lengths.{field_name} must be an integer.")
            if value <= 0:
                raise ValueError(f"eval.context_lengths.{field_name} must be greater than 0.")
        if context_lengths["start"] > context_lengths["stop"]:
            raise ValueError("eval.context_lengths.start must not be greater than stop.")

        for field_name in (
            "attention_time_fraction_at_calibration",
            "exposed_hbm_stall_fraction_at_512k",
        ):
            value = config_data["eval"][field_name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"eval.{field_name} must be a number.")
            if not 0 <= value <= 1:
                raise ValueError(f"eval.{field_name} must be between 0 and 1.")

        stall_hidden_until_context = config_data["eval"]["stall_hidden_until_context"]
        if (
            not isinstance(stall_hidden_until_context, int)
            or isinstance(stall_hidden_until_context, bool)
        ):
            raise TypeError("eval.stall_hidden_until_context must be an integer.")
        if stall_hidden_until_context < 0:
            raise ValueError("eval.stall_hidden_until_context must be greater than or equal to 0.")

        for field_name in ("output_csv", "output_plot"):
            value = config_data["eval"][field_name]
            if not isinstance(value, str):
                raise TypeError(f"eval.{field_name} must be a string.")
            if not value.strip():
                raise ValueError(f"eval.{field_name} cannot be empty.")

        memory_bandwidth_for_latency = config_data["eval"]["memory_bandwidth_for_latency"]
        if memory_bandwidth_for_latency not in ("hbm", "attention_buffer"):
            raise ValueError(
                "eval.memory_bandwidth_for_latency must be either "
                "'hbm' or 'attention_buffer'."
            )

        model = config_data["model"]
        hnlpu = config_data["hnlpu"]
        memory = config_data["memory"]
        if not isinstance(model["name"], str) or not model["name"].strip():
            raise ValueError("model.name must be a non-empty string.")
        if model["top_k_experts"] > model["num_experts"]:
            raise ValueError("model.top_k_experts must not exceed model.num_experts.")
        if model["num_weight_values"] != 2 ** model["weight_bits"]:
            raise ValueError("model.num_weight_values must equal 2 ** model.weight_bits.")
        if hnlpu["chip_grid_rows"] * hnlpu["chip_grid_cols"] != hnlpu["num_chips"]:
            raise ValueError("hnlpu chip grid dimensions must match hnlpu.num_chips.")
        if model["num_experts"] % hnlpu["num_chips"] != 0:
            raise ValueError(
                "model.num_experts must be divisible by hnlpu.num_chips."
            )
        if hnlpu["pipeline_slots"] != model["num_layers"] * hnlpu["stages_per_layer"]:
            raise ValueError(
                "hnlpu.pipeline_slots must equal model.num_layers multiplied by "
                "hnlpu.stages_per_layer."
            )
        if hnlpu["max_batch_size"] > hnlpu["pipeline_slots"]:
            raise ValueError("hnlpu.max_batch_size must not exceed hnlpu.pipeline_slots.")

        attention_buffer_size_byte = int(memory["attention_buffer_mb_per_chip"] * 1_000_000)
        attention_buffer_bank_size_byte = int(memory["attention_buffer_bank_kb"] * 1_000)
        bank_capacity_byte = memory["attention_buffer_banks_per_chip"] * attention_buffer_bank_size_byte
        if attention_buffer_size_byte != bank_capacity_byte:
            raise ValueError(
                "Attention Buffer capacity must equal bank count multiplied by "
                "bank size."
            )

        self.yaml_path = config_path
        self.data = config_data
        self.model = model
        self.hnlpu = hnlpu
        self.vex = config_data["vex"]
        self.memory = memory
        self.interconnect = config_data["interconnect"]
        self.eval = config_data["eval"]

        # Normalize units once so simulation modules use a consistent convention.
        self.clock_frequency_hz = int(hnlpu["clock_GHz"] * 1_000_000_000)
        self.attention_buffer_size_byte = attention_buffer_size_byte
        self.attention_buffer_bank_size_byte = attention_buffer_bank_size_byte
        self.attention_buffer_bandwidth_byte_per_s = int(memory["attention_buffer_bandwidth_TBps_per_chip"] * 1_000_000_000_000)
        self.hbm_size_byte = int(memory["hbm_capacity_GB_per_chip"] * 1_000_000_000)
        self.hbm_bandwidth_byte_per_s = int(memory["hbm_bandwidth_GBps_per_chip"] * 1_000_000_000)
        self.cxl_bandwidth_byte_per_s = int(self.interconnect["cxl_bandwidth_GBps_per_link"] * 1_000_000_000)
        self.cxl_latency_s = self.interconnect["cxl_latency_ns"] * 1e-9
