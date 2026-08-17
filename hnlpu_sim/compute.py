class ComputeTask:
    def __init__(self, request_id, task_type, workload):
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

        self.request_id = request_id
        self.task_type = task_type
        self.workload = workload.copy()

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

# HNArray executes operations involving fixed pretrained weights. The current
# simulator uses a coarse fixed latency and does not model HN microarchitecture.
class HNArray:
    def __init__(self, fixed_latency_cycles):
        if (
            not isinstance(fixed_latency_cycles, int)
            or isinstance(fixed_latency_cycles, bool)
        ):
            raise TypeError("fixed_latency_cycles must be an integer.")
        if fixed_latency_cycles <= 0:
            raise ValueError("fixed_latency_cycles must be greater than 0.")

        self.fixed_latency_cycles = fixed_latency_cycles
        self.busy_until_cycle = 0

    def execute(self, task, request_cycle):
        if not isinstance(task, ComputeTask):
            raise TypeError("task must be a ComputeTask.")
        if not isinstance(request_cycle, int) or isinstance(request_cycle, bool):
            raise TypeError("request_cycle must be an integer.")
        if request_cycle < 0:
            raise ValueError("request_cycle must be greater than or equal to 0.")

        if task.task_type not in ("projection", "unembedding"):
            raise ValueError(f"HNArray does not support task_type({task.task_type}).")

        # Simulator-level coarse model, not a physical latency from the paper.
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
        }


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

    hn_array = HNArray(fixed_latency_cycles = 10)
    projection_task = ComputeTask(
        request_id = "request-0",
        task_type = "projection",
        workload = {"num_tokens": 32},
    )
    projection_result = hn_array.execute(projection_task, request_cycle = 5)

    assert projection_result["start_cycle"] == 5
    assert projection_result["service_cycles"] == 10
    assert projection_result["finish_cycle"] == 15

    unembedding_task = ComputeTask(
        request_id = "request-0",
        task_type = "unembedding",
        workload = {"num_tokens": 1},
    )
    unembedding_result = hn_array.execute(
        unembedding_task,
        request_cycle = 6,
    )
    assert unembedding_result["start_cycle"] == projection_result["finish_cycle"]
    assert unembedding_result["wait_cycles"] == 9
    assert unembedding_result["service_cycles"] == 10
    assert unembedding_result["finish_cycle"] == 25

    moe_task = ComputeTask(
        request_id = "request-0",
        task_type = "moe",
        workload = {},
    )
    busy_until_cycle_before = hn_array.busy_until_cycle
    try:
        hn_array.execute(moe_task, request_cycle = 0)
    except ValueError:
        pass
    else:
        raise AssertionError("HNArray did not reject task_type(moe).")
    assert hn_array.busy_until_cycle == busy_until_cycle_before

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
