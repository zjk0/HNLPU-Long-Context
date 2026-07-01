import numpy as np
import matplotlib.pyplot as plt
import yaml
import csv

'''
@brief: parse the parameters from yaml file.
@param: param_yaml_path: the path to the yaml file.
@return: a dictionary of parameters in the yaml file.
'''
def parse_parameters(param_yaml_path):
    with open(param_yaml_path, "r", encoding="utf-8") as f:
        parameters = yaml.safe_load(f)
        
    return parameters

def throughput_eval(
    context_length,
    parameters,
    batch_size=None,
    calibration_context_length=None,
    reported_throughput=None,
):
    """Estimate decode throughput as context length grows.

    This is a simplified calibrated model, not the paper's cycle simulator.
    The fixed non-attention part is calibrated from the reported 2K-context
    throughput, while attention and HBM stalls grow with the KV length.
    """
    hnlpu = parameters["hnlpu"]
    eval_config = parameters["eval"]

    batch_size = batch_size or hnlpu["max_batch_size"]
    calibration_context_length = (
        calibration_context_length or hnlpu["calibration_context_length"]
    )
    reported_throughput = (
        reported_throughput or hnlpu["reported_throughput_tokens_per_s_at_2k"]
    )

    reported_step_time = batch_size / reported_throughput
    calibration_attention_fraction = eval_config[
        "attention_time_fraction_at_calibration"
    ]
    calibration_attention_time = (
        reported_step_time * calibration_attention_fraction
    )
    base_step_time = reported_step_time - calibration_attention_time
    attention_time = calibration_attention_time * (
        context_length / calibration_context_length
    )

    memory_stall = _exposed_memory_stall(
        context_length,
        base_step_time + attention_time,
        eval_config["stall_hidden_until_context"],
        eval_config["exposed_hbm_stall_fraction_at_512k"],
    )

    step_time = base_step_time + attention_time + memory_stall
    return batch_size / step_time

def kv_cache_size_eval(context_length, num_heads, head_dim, num_layers, bytes_per_param, batch_size):
    return 2 * context_length * num_heads * head_dim * num_layers * bytes_per_param * batch_size

def memory_access_latency_eval(
    kv_cache,
    memory_bandwidth,
    buffer_capacity=0,
    fixed_latency_ns=0,
    stall_fraction_when_spilling=1.0,
):
    """Return memory access latency/stall in seconds.

    Args:
        kv_cache: KV bytes that must be available for one decode step.
        memory_bandwidth: aggregate bandwidth in GB/s, or bytes/s if the value
            is already larger than 1e6.
        buffer_capacity: bytes that fit in the on-chip Attention Buffer.
        fixed_latency_ns: optional fixed latency term.
        stall_fraction_when_spilling: fraction of spilled traffic exposed as a
            stall after double buffering. Use 1.0 for raw latency.
    """
    bandwidth_bytes_per_s = (
        memory_bandwidth if memory_bandwidth > 1e6 else memory_bandwidth * 1e9
    )
    spilled_bytes = max(kv_cache - buffer_capacity, 0)
    transfer_time = spilled_bytes / bandwidth_bytes_per_s
    fixed_time = fixed_latency_ns * 1e-9 if spilled_bytes > 0 else 0
    return transfer_time * stall_fraction_when_spilling + fixed_time

def attention_compute_cost_eval(context_length, num_heads, head_dim, num_layers, batch_size=1):
    return 4 * context_length * num_heads * head_dim * num_layers * batch_size

def _exposed_memory_stall(
    context_length,
    non_stall_step_time,
    stall_hidden_until_context,
    exposed_hbm_stall_fraction_at_512k,
):
    if context_length <= stall_hidden_until_context:
        return 0.0

    ratio = exposed_hbm_stall_fraction_at_512k * (
        (context_length - stall_hidden_until_context)
        / max(524288 - stall_hidden_until_context, 1)
    )
    ratio = min(max(ratio, 0.0), 0.95)
    return non_stall_step_time * ratio / (1 - ratio)

def _context_lengths(eval_config):
    start = eval_config["context_lengths"]["start"]
    stop = eval_config["context_lengths"]["stop"]
    num = eval_config["context_lengths"]["num"]
    return np.unique(np.round(np.geomspace(start, stop, num)).astype(int))

def _bytes_to_gib(value):
    return value / 1024**3

def _seconds_to_ms(value):
    return value * 1e3

def run_evaluation(parameters):
    model = parameters["model"]
    hnlpu = parameters["hnlpu"]
    memory = parameters["memory"]
    eval_config = parameters["eval"]
    batch_size = hnlpu["max_batch_size"]
    contexts = _context_lengths(eval_config)

    rows = []
    for context_length in contexts:
        kv_cache = kv_cache_size_eval(
            context_length,
            model["num_kv_heads"],
            model["head_dim"],
            model["num_layers"],
            memory["kv_dtype_bytes"],
            batch_size,
        )
        attention_cost = attention_compute_cost_eval(
            context_length,
            model["num_q_heads"],
            model["head_dim"],
            model["num_layers"],
            batch_size,
        )
        hbm_latency = memory_access_latency_eval(
            kv_cache,
            memory["hbm_bandwidth_GBps_per_chip"] * hnlpu["num_chips"],
            buffer_capacity=memory["attention_buffer_mb_per_chip"]
            * 1024**2
            * hnlpu["num_chips"],
        )
        throughput = throughput_eval(context_length, parameters, batch_size=batch_size)

        rows.append(
            {
                "context_length": context_length,
                "throughput_tokens_per_s": throughput,
                "kv_cache_GiB": _bytes_to_gib(kv_cache),
                "memory_access_latency_ms": _seconds_to_ms(hbm_latency),
                "attention_compute_cost_ops": attention_cost,
            }
        )

    return rows

def save_results_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def plot_results(rows, output_path):
    contexts = np.array([row["context_length"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("throughput_tokens_per_s", "Throughput (tokens/s)"),
        ("kv_cache_GiB", "KV cache size (GiB)"),
        ("memory_access_latency_ms", "HBM access latency after SRAM spill (ms)"),
        ("attention_compute_cost_ops", "Attention compute cost (ops/token step)"),
    ]

    for ax, (key, ylabel) in zip(axes.flat, plots):
        ax.plot(contexts, [row[key] for row in rows], marker="o", linewidth=1.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.suptitle("Simplified HNLPU Performance vs. Context Length")
    fig.savefig(output_path, dpi=200)

def main():
    parameters = parse_parameters("./hnlpu_config.yaml")
    rows = run_evaluation(parameters)
    save_results_csv(rows, parameters["eval"]["output_csv"])
    plot_results(rows, parameters["eval"]["output_plot"])

    print("context_length, throughput(tokens/s), kv_cache(GiB), hbm_latency(ms), attention_ops")
    for row in rows[:: max(len(rows) // 8, 1)]:
        print(
            f"{row['context_length']}, "
            f"{row['throughput_tokens_per_s']:.2f}, "
            f"{row['kv_cache_GiB']:.2f}, "
            f"{row['memory_access_latency_ms']:.4f}, "
            f"{row['attention_compute_cost_ops']:.3e}"
        )
    print(f"Saved CSV to {parameters['eval']['output_csv']}")
    print(f"Saved plot to {parameters['eval']['output_plot']}")

if __name__ == "__main__":
    main()
