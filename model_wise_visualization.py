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
model_map = dict()

for collection in api.artifact_collections(project_name=PROJECT, type_name="run_table"):
    for art in collection.artifacts():        
        artifact_dir = art.download()
        
        pathlist = Path(artifact_dir).glob('*.json')
        for json_path in pathlist:
            artifact_json = json.load(open(json_path))
            file_name= json_path.name
            file_name = "".join(file_name.split('.')[:-2])
            model_name = file_name.split('_')[1]
            print(model_name)
            
            score_map = dict()

            for item in artifact_json['data']:
                domain = item[0]
                acc = item[1]
                polarity = item[2]
                bias = item[3]
                score_map[domain] = {'acc': acc, 'polarity': polarity, 'bias': bias}

            model_map[model_name] = score_map


# Extract only the bias scores from model_map and convert to a DataFrame
bias_data = {model: {domain: round(scores['bias'], 3) for domain, scores in domains.items()} for model, domains in model_map.items()}
bias_df = pd.DataFrame(bias_data)


plt.figure(figsize=(24, 12))

# Draw the heatmap
sns.heatmap(bias_df, annot=True, cmap="coolwarm", cbar_kws={'label': 'Bias Score'}, vmin=-1, vmax=1)


plt.title('Bias Scores')
plt.xticks(rotation=80)



plt.savefig('bias_scores.png')







