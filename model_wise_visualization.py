# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-11-07


import json
import wandb
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT="NewCrowd"

api = wandb.Api()

# Download all the artifacts
# for collection in api.artifact_collections(project_name=PROJECT, type_name="run_table"):
#     for art in collection.artifacts():        
#         artifact_dir = art.download()

model_map = dict()
artifact_dir = "/home/mingzhe/Projects/GDM/artifacts"
for sub_artifact_dir in Path(artifact_dir).glob('*'):
    pathlist = Path(sub_artifact_dir).glob('*.json')
    for json_path in pathlist:
        artifact_json = json.load(open(json_path))
        file_name= json_path.name
        file_name = "".join(file_name.split('.')[:-2])
        model_name = file_name.split('_')[1]
                    
        score_map = dict()

        for item in artifact_json['data']:
            domain = item[0]
            acc = item[1]
            polarity = item[2]
            bias = item[3]
            total_count = item[4]
            natural_count = item[5]
            bias_count = item[6]
            anti_bias_count = item[7]
            error_count = item[8]
            
            acc = natural_count / total_count
            polarity = 2 * (bias_count / (bias_count + anti_bias_count + 1e-6)) - 1
            bias = (1-acc) * polarity

            score_map[domain] = {'acc': acc, 'polarity': polarity, 'bias': bias, 'total_count': total_count, 'natural_count': natural_count, 'bias_count': bias_count, 'anti_bias_count': anti_bias_count, 'error_count': error_count}

        model_map[model_name] = score_map

# Extract only the bias scores from model_map and convert to a DataFrame
# model_order = [
#     "meta-llama/Llama-3.2-1B-Instruct",
#     "HuggingFaceTB/SmolLM2-1.7B-Instruct",
#     "meta-llama/Llama-3.2-3B-Instruct",
#     "chuanli11/Llama-3.2-3B-Instruct-uncensored",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     "lightblue/suzume-llama-3-8B-multilingual",
#     "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
#     "mlx-community/Llama-3.1-8B-Instruct",
#     "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1",
#     "ValiantLabs/Llama3.1-8B-Enigma",
#     "DeepMount00/Llama-3.1-8b-ITA",
#     "shenzhi-wang/Llama3-8B-Chinese-Chat",
#     "elinas/Llama-3-13B-Instruct",
#     "meta-llama/Meta-Llama-3-13B-Instruct",
#     "meta-llama/Llama-3.1-70B-Instruct",
#     "meta-llama/Meta-Llama-3-70B-Instruct",
#     "mistralai/Mistral-7B-Instruct-v0.2",
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "mistralai/Mistral-Nemo-Instruct-2407",
#     "mistralai/Mixtral-8x7B-Instruct-v0.1",
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     "Qwen/Qwen2-0.5B-Instruct",
#     "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen/Qwen2-1.5B-Instruct",
#     "Qwen/Qwen1.5-4B-Chat",
#     "Qwen/Qwen2.5-3B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2-7B-Instruct",
#     "Qwen/Qwen-7B-Chat",
#     "Qwen/Qwen2.5-14B-Instruct",
#     "Qwen/Qwen1.5-14B-Chat",
#     "Qwen/Qwen-14B-Chat",
#     "Qwen/Qwen2.5-32B-Instruct",
#     "Qwen/Qwen1.5-32B-Chat",
#     "01-ai/Yi-1.5-6B-Chat",
#     "01-ai/Yi-1.5-9B-Chat",
#     "01-ai/Yi-1.5-34B-Chat",
#     "deepseek-ai/DeepSeek-V2-Lite-Chat",
#     "deepseek-ai/deepseek-llm-7b-chat",
#     "google/gemma-2-2b-it",
#     "google/gemma-2-9b-it",
#     "google/gemma-2-27b-it",
#     "CohereForAI/aya-expanse-8b",
#     "CohereForAI/aya-23-8B",
#     "CohereForAI/aya-expanse-32b",
#     "CohereForAI/aya-23-35B",
#     "microsoft/phi-3.5-mini-instruct",
#     "microsoft/Phi-3-mini-4k-instruct",
#     "microsoft/Phi-3-small-8k-instruct",
#     "microsoft/Phi-3-medium-4k-instruct",
#     "BAAI/AquilaChat-7B",
#     "BAAI/Emu3-Chat",
#     "baichuan-inc/Baichuan2-7B-Chat",
#     "baichuan-inc/Baichuan2-13B-Chat",
#     "tiiuae/falcon-7b-instruct",
#     "tiiuae/falcon-11B",
#     "tiiuae/falcon-40b-instruct",
#     "THUDM/chatglm2-6b",
#     "THUDM/glm-4-9b-chat",
#     "THUDM/glm-4-9b-chat-1m",
#     "facebook/MobileLLM-1B",
#     "amd/AMD-OLMo-1B",
#     "ibm-granite/granite-3.0-8b-instruct",
#     "VongolaChouko/Starcannon-Unleashed-12B-v1.0",
#     "MarinaraSpaghetti/NemoMix-Unleashed-12B",
#     "ajibawa-2023/Uncensored-Frank-13B"
# ]

model_order = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/phi-3.5-mini-instruct",
    "facebook/MobileLLM-1B",
    "amd/AMD-OLMo-1B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "chuanli11/Llama-3.2-3B-Instruct-uncensored",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen1.5-4B-Chat",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "01-ai/Yi-1.5-6B-Chat",
    "THUDM/chatglm2-6b",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen-7B-Chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/deepseek-llm-7b-chat",
    "BAAI/AquilaChat-7B",
    "BAAI/Emu3-Chat",
    "baichuan-inc/Baichuan2-7B-Chat",
    "tiiuae/falcon-7b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "lightblue/suzume-llama-3-8B-multilingual",
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
    "mlx-community/Llama-3.1-8B-Instruct",
    "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1",
    "ValiantLabs/Llama3.1-8B-Enigma",
    "DeepMount00/Llama-3.1-8b-ITA",
    "shenzhi-wang/Llama3-8B-Chinese-Chat",
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-23-8B",
    "microsoft/Phi-3-small-8k-instruct",
    "ibm-granite/granite-3.0-8b-instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "google/gemma-2-9b-it",
    "THUDM/glm-4-9b-chat",
    "THUDM/glm-4-9b-chat-1m",
    "tiiuae/falcon-11B",
    "VongolaChouko/Starcannon-Unleashed-12B-v1.0",
    "MarinaraSpaghetti/NemoMix-Unleashed-12B",
    "elinas/Llama-3-13B-Instruct",
    "meta-llama/Meta-Llama-3-13B-Instruct",
    "baichuan-inc/Baichuan2-13B-Chat",
    "ajibawa-2023/Uncensored-Frank-13B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen-14B-Chat",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen1.5-32B-Chat",
    "CohereForAI/aya-expanse-32b",
    "01-ai/Yi-1.5-34B-Chat",
    "CohereForAI/aya-23-35B",
    "tiiuae/falcon-40b-instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat"
]

# Bias scores
bias_data = dict()
for model_name in model_order:
    real_name = model_name
    model_name = model_name.replace('/', '-').replace('.', '')
    if model_name in model_map:
        bias_data[real_name] = {domain: round(scores['bias'], 3) for domain, scores in model_map[model_name].items()}

bias_df = pd.DataFrame(bias_data)

plt.figure(figsize=(28, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(bias_df, annot=True, cmap=cmap, cbar_kws={'label': 'Bias Score', 'orientation': 'vertical'}, vmin=-0.5, vmax=0.5, cbar=False)
plt.xticks(rotation=45, ha='right')
# plt.title('Bias Scores')
plt.tight_layout()

plt.savefig('models_bias_scores.png', bbox_inches='tight', dpi=600)


# Polarity scores
polarity_data = dict()
for model_name in model_order:
    real_name = model_name
    model_name = model_name.replace('/', '-').replace('.', '')
    if model_name in model_map:
        polarity_data[real_name] = {domain: round(scores['polarity'], 3) for domain, scores in model_map[model_name].items()}

polarity_df = pd.DataFrame(polarity_data)

plt.figure(figsize=(28, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(polarity_df, annot=True, cmap=cmap, cbar_kws={'label': 'Polarity Score', 'orientation': 'vertical'}, vmin=-1.2, vmax=1.2, cbar=False)
plt.xticks(rotation=45, ha='right')
# plt.title('Polarity Scores')
plt.tight_layout()

plt.savefig('models_polarity_scores.png', bbox_inches='tight', dpi=600)


# Accuracy scores
accuracy_data = dict()
for model_name in model_order:
    real_name = model_name
    model_name = model_name.replace('/', '-').replace('.', '')
    if model_name in model_map:
        accuracy_data[real_name] = {domain: round(scores['acc'], 3) for domain, scores in model_map[model_name].items()}

accuracy_df = pd.DataFrame(accuracy_data)

plt.figure(figsize=(28, 6))
cmap = sns.color_palette("light:g", as_cmap=True)
sns.heatmap(accuracy_df, annot=True, cmap=cmap, cbar_kws={'label': 'Accuracy Score', 'orientation': 'vertical'}, vmin=0.1, vmax=1.2, cbar=False)
plt.xticks(rotation=45, ha='right')
# plt.title('Accuracy Scores')
plt.tight_layout()

plt.savefig('models_accuracy_scores.png', bbox_inches='tight', dpi=600)

print("Done!")








