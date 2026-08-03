class Request:
    def __init__(
        self,
        request_id,
        input_token_num,
        output_token_num,
        arrival_cycle,
    ):
        self.request_id = request_id
        self.input_token_num = input_token_num
        self.output_token_num = output_token_num
        self.arrival_cycle = arrival_cycle

        self.generated_token_num = 0
        self.current_token_position = input_token_num
        self.status = "waiting"
        self.phase = "prefill"
        self.start_cycle = None
        self.finish_cycle = None
