class ComputeTask:
    def __init__(
        self,
        request_id,
        task_type,
        workload,
        layer_id = None,
        weight_type = None,
        expert_id = None,
    ):
        if request_id is None or (
            isinstance(request_id, str) and not request_id.strip()
        ):
            raise ValueError("request_id must not be empty.")
        try:
            hash(request_id)
        except TypeError as exc:
            raise TypeError("request_id must be hashable.") from exc

        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string.")
        if not task_type.strip():
            raise ValueError("task_type must not be empty.")

        if not isinstance(workload, dict):
            raise TypeError("workload must be a dict.")

        if layer_id is not None:
            if not isinstance(layer_id, int) or isinstance(layer_id, bool):
                raise TypeError("layer_id must be an integer or None.")
            if layer_id < 0:
                raise ValueError("layer_id must be greater than or equal to 0.")

        if weight_type is not None:
            if not isinstance(weight_type, str):
                raise TypeError("weight_type must be a string or None.")
            if not weight_type.strip():
                raise ValueError("weight_type must not be empty.")

        if expert_id is not None:
            if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                raise TypeError("expert_id must be an integer or None.")
            if expert_id < 0:
                raise ValueError("expert_id must be greater than or equal to 0.")

        self.request_id = request_id
        self.task_type = task_type
        self.workload = workload.copy()

        # Identifies the Transformer layer that owns this task.
        self.layer_id = layer_id
        # Identifies the fixed-weight operation used for HNArray routing.
        self.weight_type = weight_type
        # Identifies the MoE expert associated with expert-specific computation.
        self.expert_id = expert_id

# The physical VEX also handles RMSNorm, SwiGLU, softmax, residual
# addition, and sampling. Only attention has a timing model here because
# the paper does not provide cycle-level parameters for the other operations.
class VEX:
    def __init__(self, cached_kv_heads_per_cycle = 32):
        if (
            not isinstance(cached_kv_heads_per_cycle, int)
            or isinstance(cached_kv_heads_per_cycle, bool)
        ):
            raise TypeError("cached_kv_heads_per_cycle must be an integer.")
        if cached_kv_heads_per_cycle <= 0:
            raise ValueError("cached_kv_heads_per_cycle must be greater than 0.")

        self.cached_kv_heads_per_cycle = cached_kv_heads_per_cycle
        self.busy_until_cycle = 0

    def execute(self, task, request_cycle):
        if not isinstance(task, ComputeTask):
            raise TypeError("task must be a ComputeTask.")
        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")

        vex_task_types = (
            "attention",
            "rmsnorm",
            "swiglu",
            "softmax",
            "residual",
            "sampling",
        )
        if task.task_type not in vex_task_types:
            raise ValueError(f"VEX does not support task_type({task.task_type}).")
        if task.task_type != "attention":
            raise NotImplementedError(
                f"{task.task_type} is supported by the physical HNLPU VEX, "
                "but its timing model is not implemented in the current "
                "simulator."
            )

        for field_name in ("q_length", "kv_length"):
            if field_name not in task.workload:
                raise KeyError(f"attention workload must contain {field_name}.")

            field_value = task.workload[field_name]
            if not isinstance(field_value, int) or isinstance(field_value, bool):
                raise TypeError(f"attention workload {field_name} must be an integer.")
            if field_value <= 0:
                raise ValueError(f"attention workload {field_name} must be greater than 0.")

        q_length = task.workload["q_length"]
        kv_length = task.workload["kv_length"]

        # First-version equivalent attention workload model. This value is
        # not the physical cached KV-head count defined by the paper.
        equivalent_cached_kv_head_work = q_length * kv_length
        service_cycles = max(
            1, 
            (equivalent_cached_kv_head_work + self.cached_kv_heads_per_cycle - 1) // self.cached_kv_heads_per_cycle
        )

        start_cycle = max(request_cycle, self.busy_until_cycle)
        wait_cycles = start_cycle - request_cycle
        finish_cycle = start_cycle + service_cycles
        total_latency_cycles = finish_cycle - request_cycle

        self.busy_until_cycle = finish_cycle

        return {
            "request_cycle": request_cycle,
            "start_cycle": start_cycle,
            "finish_cycle": finish_cycle,
            "wait_cycles": wait_cycles,
            "service_cycles": service_cycles,
            "total_latency_cycles": total_latency_cycles,
            "compute_workload": {
                "q_length": q_length,
                "kv_length": kv_length,
                "equivalent_cached_kv_head_work": equivalent_cached_kv_head_work,
            },
        }

class HNArray:
    NON_EXPERT_WEIGHT_TYPES = ("q", "k", "v", "xo", "router")
    EXPERT_WEIGHT_TYPES = ("up", "gate", "down")
    SUPPORTED_WEIGHT_TYPES = NON_EXPERT_WEIGHT_TYPES + EXPERT_WEIGHT_TYPES

    def __init__(self, layer_num, expert_ids, weight_type_latency):
        if not isinstance(layer_num, int) or isinstance(layer_num, bool):
            raise TypeError("layer_num must be an integer.")
        if layer_num <= 0:
            raise ValueError("layer_num must be greater than 0.")

        if not isinstance(expert_ids, list):
            raise TypeError("expert_ids must be a list.")
        for expert_id in expert_ids:
            if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                raise TypeError("Every expert_id must be an integer.")
            if expert_id < 0:
                raise ValueError("Every expert_id must be greater than or equal to 0.")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("expert_ids must not contain duplicates.")

        if not isinstance(weight_type_latency, dict):
            raise TypeError("weight_type_latency must be a dictionary.")
        missing_weight_types = [
            weight_type
            for weight_type in self.SUPPORTED_WEIGHT_TYPES
            if weight_type not in weight_type_latency
        ]
        if missing_weight_types:
            raise KeyError(
                "weight_type_latency is missing weight types: "
                + ", ".join(missing_weight_types)
            )

        for weight_type in self.SUPPORTED_WEIGHT_TYPES:
            latency = weight_type_latency[weight_type]
            if not isinstance(latency, int) or isinstance(latency, bool):
                raise TypeError(f"weight_type_latency[{weight_type!r}] must be an integer.")
            if latency <= 0:
                raise ValueError(f"weight_type_latency[{weight_type!r}] must be greater than 0.")

        self.layer_num = layer_num
        self.expert_ids = expert_ids.copy()
        self.weight_type_latency = weight_type_latency.copy()
        self.units = {}

        for layer_id in range(layer_num):
            for weight_type in self.NON_EXPERT_WEIGHT_TYPES:
                unit_key = (layer_id, weight_type, None)
                self.units[unit_key] = HNUnit(
                    layer_id = layer_id,
                    weight_type = weight_type,
                    fixed_latency_cycles = weight_type_latency[weight_type],
                )

            for weight_type in self.EXPERT_WEIGHT_TYPES:
                for expert_id in expert_ids:
                    unit_key = (layer_id, weight_type, expert_id)
                    self.units[unit_key] = HNUnit(
                        layer_id = layer_id,
                        weight_type = weight_type,
                        fixed_latency_cycles = weight_type_latency[weight_type],
                        expert_id = expert_id,
                    )

    def execute(
        self,
        task,
        layer_id,
        weight_type,
        request_cycle,
        expert_id = None,
    ):
        if not isinstance(layer_id, int) or isinstance(layer_id, bool):
            raise TypeError("layer_id must be an integer.")
        if not 0 <= layer_id < self.layer_num:
            raise ValueError(
                f"layer_id must be between 0 and {self.layer_num - 1}."
            )

        if not isinstance(weight_type, str):
            raise TypeError("weight_type must be a string.")
        if not weight_type.strip():
            raise ValueError("weight_type must not be empty.")
        if weight_type not in self.SUPPORTED_WEIGHT_TYPES:
            raise ValueError(f"HNArray does not support weight_type({weight_type}).")

        if expert_id is not None:
            if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                raise TypeError("expert_id must be an integer or None.")
            if expert_id < 0:
                raise ValueError("expert_id must be greater than or equal to 0.")

        if weight_type in self.NON_EXPERT_WEIGHT_TYPES and expert_id is not None:
            raise ValueError(
                f"weight_type({weight_type}) requires expert_id to be None."
            )
        if weight_type in self.EXPERT_WEIGHT_TYPES and expert_id is None:
            raise ValueError(f"weight_type({weight_type}) requires an expert_id.")

        unit_key = (layer_id, weight_type, expert_id)
        if unit_key not in self.units:
            raise KeyError(f"HNUnit does not exist for key {unit_key}.")

        return self.units[unit_key].execute(task, request_cycle)


class HNUnit:
    def __init__(
        self,
        layer_id,
        weight_type,
        fixed_latency_cycles,
        expert_id = None,
    ):
        if not isinstance(layer_id, int) or isinstance(layer_id, bool):
            raise TypeError("layer_id must be an integer.")
        if layer_id < 0:
            raise ValueError("layer_id must be greater than or equal to 0.")

        if not isinstance(weight_type, str):
            raise TypeError("weight_type must be a string.")
        if not weight_type.strip():
            raise ValueError("weight_type must not be empty.")

        if (
            not isinstance(fixed_latency_cycles, int)
            or isinstance(fixed_latency_cycles, bool)
        ):
            raise TypeError("fixed_latency_cycles must be an integer.")
        if fixed_latency_cycles <= 0:
            raise ValueError("fixed_latency_cycles must be greater than 0.")

        if expert_id is not None:
            if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                raise TypeError("expert_id must be an integer or None.")
            if expert_id < 0:
                raise ValueError("expert_id must be greater than or equal to 0.")

        self.layer_id = layer_id
        self.weight_type = weight_type
        self.fixed_latency_cycles = fixed_latency_cycles
        self.expert_id = expert_id
        self.busy_until_cycle = 0

    def execute(self, task, request_cycle):
        if not isinstance(task, ComputeTask):
            raise TypeError("task must be a ComputeTask.")
        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")

        service_cycles = self.fixed_latency_cycles
        start_cycle = max(request_cycle, self.busy_until_cycle)
        wait_cycles = start_cycle - request_cycle
        finish_cycle = start_cycle + service_cycles
        total_latency_cycles = finish_cycle - request_cycle

        self.busy_until_cycle = finish_cycle

        return {
            "request_cycle": request_cycle,
            "start_cycle": start_cycle,
            "finish_cycle": finish_cycle,
            "wait_cycles": wait_cycles,
            "service_cycles": service_cycles,
            "total_latency_cycles": total_latency_cycles,
            "compute_workload": task.workload.copy(),
            "layer_id": self.layer_id,
            "weight_type": self.weight_type,
            "expert_id": self.expert_id,
        }


if __name__ == "__main__":
    vex = VEX()
    attention_task = ComputeTask(
        request_id = "request-0",
        task_type = "attention",
        workload = {
            "q_length": 1,
            "kv_length": 64,
        },
    )
    first_vex_result = vex.execute(attention_task, request_cycle = 0)
    second_vex_result = vex.execute(attention_task, request_cycle = 0)

    assert first_vex_result["service_cycles"] == 2
    assert first_vex_result["wait_cycles"] == 0
    assert second_vex_result["start_cycle"] == first_vex_result["finish_cycle"]
    assert second_vex_result["wait_cycles"] == first_vex_result["service_cycles"]

    for task_type in (
        "rmsnorm",
        "swiglu",
        "softmax",
        "residual",
        "sampling",
    ):
        vex_task = ComputeTask(
            request_id = "request-0",
            task_type = task_type,
            workload = {},
        )
        busy_until_cycle_before = vex.busy_until_cycle
        try:
            vex.execute(vex_task, request_cycle = 0)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(
                f"VEX did not report an unimplemented timing model for {task_type}."
            )
        assert vex.busy_until_cycle == busy_until_cycle_before

    for task_type in ("projection", "unknown", "moe"):
        invalid_vex_task = ComputeTask(
            request_id = "request-0",
            task_type = task_type,
            workload = {},
        )
        try:
            vex.execute(invalid_vex_task, request_cycle = 0)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"VEX did not reject unsupported task_type({task_type})."
            )

    weight_type_latency = {
        "q": 7,
        "k": 8,
        "v": 9,
        "xo": 10,
        "router": 11,
        "up": 12,
        "gate": 13,
        "down": 14,
    }
    hn_array = HNArray(
        layer_num = 2,
        expert_ids = [3, 7],
        weight_type_latency = weight_type_latency,
    )
    assert len(hn_array.units) == 2 * (5 + 3 * 2)
    assert (0, "q", None) in hn_array.units
    assert (0, "up", 3) in hn_array.units

    projection_task = ComputeTask(
        request_id = "request-0",
        task_type = "projection",
        workload = {"num_tokens": 1},
    )

    layer_0_q_result = hn_array.execute(
        projection_task,
        layer_id = 0,
        weight_type = "q",
        request_cycle = 5,
    )
    layer_1_q_result = hn_array.execute(
        projection_task,
        layer_id = 1,
        weight_type = "q",
        request_cycle = 5,
    )
    layer_0_k_result = hn_array.execute(
        projection_task,
        layer_id = 0,
        weight_type = "k",
        request_cycle = 5,
    )

    assert layer_0_q_result["start_cycle"] == 5
    assert layer_1_q_result["start_cycle"] == 5
    assert layer_0_k_result["start_cycle"] == 5
    assert layer_0_q_result["layer_id"] == 0
    assert layer_1_q_result["layer_id"] == 1
    assert layer_0_k_result["weight_type"] == "k"

    queued_q_result = hn_array.execute(
        projection_task,
        layer_id = 0,
        weight_type = "q",
        request_cycle = 5,
    )
    assert queued_q_result["start_cycle"] == layer_0_q_result["finish_cycle"]
    assert queued_q_result["wait_cycles"] == 7

    expert_3_result = hn_array.execute(
        projection_task,
        layer_id = 0,
        weight_type = "up",
        expert_id = 3,
        request_cycle = 5,
    )
    expert_7_result = hn_array.execute(
        projection_task,
        layer_id = 0,
        weight_type = "up",
        expert_id = 7,
        request_cycle = 5,
    )
    assert expert_3_result["start_cycle"] == 5
    assert expert_7_result["start_cycle"] == 5
    assert expert_3_result["expert_id"] == 3
    assert expert_7_result["expert_id"] == 7

    print("VEX and HNArray tests passed.")

    first_hn_unit = HNUnit(
        layer_id = 0,
        weight_type = "qkv_projection",
        fixed_latency_cycles = 7,
    )
    second_hn_unit = HNUnit(
        layer_id = 1,
        weight_type = "expert_up_projection",
        fixed_latency_cycles = 11,
        expert_id = 3,
    )
    hn_unit_task = ComputeTask(
        request_id = "request-0",
        task_type = "projection",
        workload = {"num_tokens": 1},
    )

    first_unit_result = first_hn_unit.execute(
        hn_unit_task,
        request_cycle = 5,
    )
    second_unit_result = second_hn_unit.execute(
        hn_unit_task,
        request_cycle = 5,
    )

    assert first_unit_result["start_cycle"] == 5
    assert second_unit_result["start_cycle"] == 5
    assert first_unit_result["finish_cycle"] == 12
    assert second_unit_result["finish_cycle"] == 16
    assert first_unit_result["layer_id"] == 0
    assert first_unit_result["weight_type"] == "qkv_projection"
    assert first_unit_result["expert_id"] is None
    assert second_unit_result["expert_id"] == 3

    queued_unit_result = first_hn_unit.execute(
        hn_unit_task,
        request_cycle = 5,
    )
    assert queued_unit_result["start_cycle"] == first_unit_result["finish_cycle"]
    assert queued_unit_result["wait_cycles"] == 7
    assert queued_unit_result["finish_cycle"] == 19
    assert first_hn_unit.busy_until_cycle == 19
    assert second_hn_unit.busy_until_cycle == 16

    print("HNUnit tests passed.")
