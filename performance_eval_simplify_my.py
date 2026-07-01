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

def throughput_eval():
    pass

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
    attention_buffer_bandwidth, 
    hbm_bandwidth, 
    attention_buffer_size
):
    if kv_cache <= parameters["memory"]["attention_buffer_mb_per_chip"]:
        return kv_cache / (attention_buffer_bandwidth * 1e12) * 1e3
    else:
        return attention_buffer_size

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
        ("attention compute cost", "attention compute cost (ops/s)"), 
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
        performance["kv cache size"].append(kv_cache)
        
        attention_compute_cost = attention_compute_cost_eval(
            context_length, 
            parameters["model"]["num_q_heads"], 
            parameters["model"]["head_dim"], 
            parameters["model"]["num_layers"], 
            parameters["hnlpu"]["max_batch_size"]
        )
        performance["attention compute cost"].append(attention_compute_cost)
        
if __name__ == "__main__":
    main()