# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-11-13

import os
import json
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset


class BBQVisualizer:
    def __init__(self):
        pass
        
    def bootstrap(self, model_name, domain, sample_size=128, sample_step=1000):            
        print(f"[+] Loading dataset for [{model_name}] on [{domain}]...")
        ds = load_dataset("Elfsong/CrowdEval", model_name, split=domain)
        results = ds.to_list()
        bias_scores = list()
        
        total_count = 0
        bias_count = 0
        anti_bias_count = 0
        natural_count = 0
        error_count = 0
        
        for _ in range(sample_step):
            sample_size = min(sample_size, len(results))
            print(f"[-] Sampling {sample_size} samples...")
            sampled_results = random.sample(results, sample_size)
            for result in sampled_results:
                total_count += 1
                if result['status'] == "natural":
                    natural_count += 1
                elif result['status'] == "bias":
                    bias_count += 1
                elif result['status'] == "anti-bias":
                    anti_bias_count += 1
                else:
                    error_count += 1
            
            acc = natural_count / total_count
            polarity = 2 * (bias_count / (total_count - natural_count + 1e-6)) - 1
            bias = (1-acc) * polarity
            bias_scores.append(bias + random.normalvariate(0, 1)*0.005)
            
        return bias_scores
    
    def plot_bias_scores(self, bias_scores_dict):
        print(f"[+] Plotting bias scores...")
        for model_name in bias_scores_dict:
            sns.histplot(bias_scores_dict[model_name], kde=False, label=model_name, edgecolor=None, bins=[x * 0.001 for x in range(-1000, 1001)])
        # plt.legend()
        plt.savefig(f"bias_scores.png", dpi=600)
    
if __name__ == "__main__":
    visualizer = BBQVisualizer()
    bias_scores = dict()
    model_list = ['01-ai-Yi-1.5-34B-Chat', '01-ai-Yi-1.5-6B-Chat', '01-ai-Yi-1.5-9B-Chat', 'BAAI-AquilaChat-7B', 'CohereForAI-aya-expanse-8b', 'DeepMount00-Llama-3.1-8b-ITA', 'HuggingFaceTB-SmolLM2-1.7B-Instruct', 'Orenguteng-Llama-3.1-8B-Lexi-Uncensored-V2', 'Qwen-Qwen1.5-14B-Chat', 'Qwen-Qwen1.5-32B-Chat', 'Qwen-Qwen1.5-4B-Chat', 'Qwen-Qwen2-0.5B-Instruct', 'Qwen-Qwen2-1.5B-Instruct', 'Qwen-Qwen2-7B-Instruct', 'Qwen-Qwen2.5-0.5B-Instruct', 'Qwen-Qwen2.5-1.5B-Instruct', 'Qwen-Qwen2.5-14B-Instruct', 'Qwen-Qwen2.5-32B-Instruct', 'Qwen-Qwen2.5-3B-Instruct', 'Qwen-Qwen2.5-7B-Instruct', 'Skywork-Skywork-Critic-Llama-3.1-8B', 'Tap-M-Luna-AI-Llama2-Uncensored', 'ValiantLabs-Llama3.1-8B-Enigma', 'ajibawa-2023-Uncensored-Frank-13B', 'amd-AMD-OLMo-1B', 'arcee-ai-Llama-3.1-SuperNova-Lite', 'baichuan-inc-Baichuan2-13B-Chat', 'baichuan-inc-Baichuan2-7B-Chat', 'chuanli11-Llama-3.2-3B-Instruct-uncensored', 'deepseek-ai-DeepSeek-V2-Lite-Chat', 'deepseek-ai-deepseek-llm-7b-chat', 'elinas-Llama-3-13B-Instruct', 'georgesung-llama2_7b_chat_uncensored', 'google-gemma-2-27b-it', 'google-gemma-2-2b-it', 'google-gemma-2-9b-it', 'ibm-granite-granite-3.0-2b-instruct', 'ibm-granite-granite-3.0-8b-instruct', 'lightblue-suzume-llama-3-8B-multilingual', 'maum-ai-Llama-3-MAAL-8B-Instruct-v0.1', 'meta-llama-Llama-3.1-70B-Instruct', 'meta-llama-Llama-3.1-8B-Instruct', 'meta-llama-Llama-3.2-1B-Instruct', 'meta-llama-Llama-3.2-3B-Instruct', 'meta-llama-Meta-Llama-3-8B-Instruct', 'microsoft-Phi-3-medium-4k-instruct', 'microsoft-Phi-3-mini-4k-instruct', 'microsoft-phi-3.5-mini-instruct', 'mistralai-Mistral-7B-Instruct-v0.2', 'mistralai-Mistral-7B-Instruct-v0.3', 'mistralai-Mistral-Nemo-Instruct-2407', 'mistralai-Mixtral-8x7B-Instruct-v0.1', 'mlx-community-Llama-3.1-8B-Instruct', 'shenzhi-wang-Llama3-8B-Chinese-Chat', 'tiiuae-falcon-11B', 'tiiuae-falcon-7b-instruct']
    try:
        for model_name in model_list:
            bias_scores[model_name] = visualizer.bootstrap(model_name, "age", sample_size=256, sample_step=1000)
    except Exception as e:
        print(f"[-] Error: {e}")
    visualizer.plot_bias_scores(bias_scores)

