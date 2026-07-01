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

def kv_cache_size_eval(context_length, num_heads, head_dim, num_layers, bytes_per_param):
    return 2 * context_length * num_heads * head_dim * num_layers * bytes_per_param

def memory_access_latency_eval(kv_cache, memory_bandwidth):
    pass

def attention_compute_cost_eval(context_length, num_heads, head_dim, num_layers):
    return 4 * context_length * num_heads * head_dim * num_layers

def main():
    parameters = parse_parameters("./hnlpu_config.yaml")
    print(parameters)

if __name__ == "__main__":
    main()