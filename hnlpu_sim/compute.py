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

class VEX:
    def __init__(self):
        pass

class HNArray:
    def __init__(self):
        pass
