# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-10-28

import os
os.environ['HF_HUB_CACHE'] = '/mnt/data'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

import wandb
import argparse
import crowd_eval
import crowd_agent

parser = argparse.ArgumentParser(description="Run the BBQEvaluator with specified agent.")
parser.add_argument('--model-type', type=str, required=True, help='Type of the model to use')
parser.add_argument('--model-name', type=str, required=True, help='Name of the model to use')
parser.add_argument('--domain', type=str, default="all", help='Domain to evaluate')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('--num-samples', type=int, default=1024, help='Number of samples')
args = parser.parse_args()

wandb_run = wandb.init(
    name=args.model_name.replace("/", "-"),
    project="NewCrowd",
    config={
        "model_type": args.model_type, 
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples
    }
)

agent_manager = crowd_agent.AgentManager()
agent = agent_manager.get_agent(args.model_type, args.model_name)
evaluator = crowd_eval.BBQEvaluator(agent, args.batch_size)
result_table = wandb.Table(columns=["domain", "accuracy_score", "polarity_score", "bias_score", "total_count", "natural_count", "bias_count", "anti_bias_count", "error_count"])

domains = ["age", "gender_identity", "disability_status", "nationality", "race_ethnicity", "religion", "ses", "sexual_orientation"]
for index, domain in enumerate(domains):
    print(f"[+] Evaluating [{domain}] [{index + 1}/{len(domains)}]...")
    scores = evaluator.evaluate(domain, args.num_samples)
    
    print(f"[{domain}] accuracy_score: {scores[0]:.4f}, polarity_score: {scores[1]:.4f}, bias_score: {scores[2]:.4f}, total_count: {scores[3]}, natural_count: {scores[4]}, bias_count: {scores[5]}, anti_bias_count: {scores[6]}, error_count: {scores[7]}")
    result_table.add_data(domain, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], scores[6], scores[7])
wandb_run.log({f"results_{args.model_name.replace('/', '-')}": result_table})
