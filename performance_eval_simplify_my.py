import numpy as np
import matplotlib.pyplot as plt
import yaml

'''
@brief: parse the parameters from yaml file.
@param: param_yaml_path: the path to the yaml file.
@return: a dictionary of parameters in the yaml file.
'''
def parse_parameters(param_yaml_path):
    with open(param_yaml_path, "r", encoding="utf-8") as f:
        parameters = yaml.safe_load(f)
        
    return parameters

def additional_memory_stall_eval(
    context_length, 
    stall_hidden_until_context, 
    exposed_hbm_stall_fraction_at_512k, 
    non_stall_time
):
    if context_length <= stall_hidden_until_context:
        return 0.0
    else:
        ratio = exposed_hbm_stall_fraction_at_512k * ((context_length - stall_hidden_until_context) / (524288 - stall_hidden_until_context))
        return non_stall_time * (ratio / (1 - ratio))

def throughput_eval(
    context_length, 
    parameters,
    batch_size = None
):
    batch_size = parameters["hnlpu"]["max_batch_size"] if batch_size is None else batch_size
    calibration_context_length = parameters["hnlpu"]["calibration_context_length"]
    reported_throughput = parameters["hnlpu"]["reported_throughput_tokens_per_s_at_2k"]
    reported_time = batch_size / reported_throughput
    
    calibration_attention_time_fraction = parameters["eval"]["attention_time_fraction_at_calibration"]
    calibration_attention_time = reported_time * calibration_attention_time_fraction
    other_time = reported_time - calibration_attention_time
    attention_time = calibration_attention_time * (context_length / calibration_context_length) 
    
    additional_memory_stall = additional_memory_stall_eval(
        context_length, 
        parameters["eval"]["stall_hidden_until_context"], 
        parameters["eval"]["exposed_hbm_stall_fraction_at_512k"], 
        other_time + attention_time
    )
    
    total_time = other_time + attention_time + additional_memory_stall
    return batch_size / total_time

def kv_cache_size_eval(
    context_length, 
    num_heads, 
    head_dim, 
    num_layers, 
    bytes_per_param, 
    batch_size
):
    return 2 * context_length * num_heads * head_dim * num_layers * bytes_per_param * batch_size

def memory_access_latency_eval(
    parameters, 
    kv_cache, 
):
    attention_buffer_bandwidth = parameters["memory"]["attention_buffer_bandwidth_TBps_per_chip"]
    attention_buffer_size = parameters["memory"]["attention_buffer_mb_per_chip"]
    kv_cache_per_chip = kv_cache / parameters["hnlpu"]["num_chips"]
    if kv_cache_per_chip <= attention_buffer_size * (1024 ** 2):
        return kv_cache_per_chip / (attention_buffer_bandwidth * 1e12) * 1e3
    else:
        hbm_bandwidth = parameters["memory"]["hbm_bandwidth_GBps_per_chip"]
        hbm_kv_cache_per_chip = kv_cache_per_chip - attention_buffer_size * (1024 ** 2)
        attention_buffer_latency = (attention_buffer_size * (1024 ** 2)) / (attention_buffer_bandwidth * 1e12) * 1e3
        hbm_latency = hbm_kv_cache_per_chip / (hbm_bandwidth * 1e9) * 1e3
        return attention_buffer_latency + hbm_latency

def attention_compute_cost_eval(
    context_length, 
    num_heads, 
    head_dim, 
    num_layers, 
    batch_size
):
    return 4 * context_length * num_heads * head_dim * num_layers * batch_size

def plot_performance_vs_context_length(
    performance, 
    context,
    plot_path
):
    fig, axes = plt.subplots(2, 2, figsize = (12, 8))
    
    info = [
        ("kv cache size", "kv cache size (GB)"), 
        ("memory access latency", "memory access latency (ms)"), 
        ("attention compute cost", "attention compute cost (ops)"), 
        ("throughput", "throughput (tokens/s)")
    ]
    
    for ax, (key, ylabel) in zip(axes.flat, info):
        ax.plot(context, performance[key])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle = "--", alpha = 0.35, which = "both")
        
    fig.suptitle("Simplified Performance vs Context Length")
    fig.savefig(plot_path, dpi = 200)

def main():
    parameters = parse_parameters("./hnlpu_config.yaml")
    context_start = parameters["eval"]["context_lengths"]["start"]
    context_stop = parameters["eval"]["context_lengths"]["stop"]
    context_num = parameters["eval"]["context_lengths"]["num"]
    context_lengths = np.unique(np.round(np.geomspace(context_start, context_stop, context_num)).astype(int))
    
    performance = {
        "kv cache size": [], 
        "memory access latency": [], 
        "attention compute cost": [], 
        "throughput": []
    }
    for context_length in context_lengths:
        kv_cache = kv_cache_size_eval(
            context_length, 
            parameters["model"]["num_kv_heads"], 
            parameters["model"]["head_dim"], 
            parameters["model"]["num_layers"], 
            parameters["memory"]["kv_dtype_bytes"], 
            parameters["hnlpu"]["max_batch_size"]
        )
        performance["kv cache size"].append(kv_cache / (1024 ** 3))
        
        attention_compute_cost = attention_compute_cost_eval(
            context_length, 
            parameters["model"]["num_q_heads"], 
            parameters["model"]["head_dim"], 
            parameters["model"]["num_layers"], 
            parameters["hnlpu"]["max_batch_size"]
        )
        performance["attention compute cost"].append(attention_compute_cost)
        
        memory_access_latency_ms = memory_access_latency_eval(
            parameters, 
            kv_cache
        )
        performance["memory access latency"].append(memory_access_latency_ms)
        
        throughput = throughput_eval(
            context_length, 
            parameters
        )
        performance["throughput"].append(throughput)
        
    plot_performance_vs_context_length(performance, context_lengths, "./performance_vs_context.png")
        
if __name__ == "__main__":
    main()