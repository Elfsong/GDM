

# for domain in ["age", "gender_identity", "disability_status", "nationality", "race_ethnicity", "religion", "ses", "sexual_orientation"]:

# LlamaAgent
python crowd.py --model-type LlamaAgent --model-name meta-llama/Llama-3.2-1B-Instruct --domain all
python crowd.py --model-type LlamaAgent --model-name meta-llama/Llama-3.2-3B-Instruct --domain all
python crowd.py --model-type LlamaAgent --model-name meta-llama/Llama-3.1-8B-Instruct --domain all
python crowd.py --model-type LlamaAgent --model-name meta-llama/Meta-Llama-3-8B-Instruct --domain all
python crowd.py --model-type LlamaAgent --model-name meta-llama/Llama-3.1-70B-Instruct --domain all
python crowd.py --model-type LlamaAgent --model-name meta-llama/Meta-Llama-3-70B-Instruct --domain all

# MixtralAgent
python crowd.py --model-type MixtralAgent --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --domain all
python crowd.py --model-type MixtralAgent --model-name mistralai/Mistral-7B-Instruct-v0.2 --domain all 
python crowd.py --model-type MixtralAgent --model-name mistralai/Mixtral-8x22B-Instruct-v0.1 --domain all 
python crowd.py --model-type MixtralAgent --model-name mistralai/Mistral-7B-Instruct-v0.3 --domain all 
python crowd.py --model-type MixtralAgent --model-name mistralai/Mistral-Nemo-Instruct-2407 --domain all 

# QwenAgent
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2-7B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-0.5B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-1.5B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-3B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-7B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-14B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-32B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2.5-72B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2-0.5B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2-1.5B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2-7B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen2-72B-Instruct --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen1.5-4B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen1.5-14B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen1.5-32B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen1.5-72B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen-1_8B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen-7B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen-14B-Chat --domain all
python crowd.py --model-type QwenAgent --model-name Qwen/Qwen-72B-Chat --domain all

# YiAgent
python crowd.py --model-type YiAgent --model-name 01-ai/Yi-1.5-6B-Chat --domain all
python crowd.py --model-type YiAgent --model-name 01-ai/Yi-1.5-9B-Chat --domain all
python crowd.py --model-type YiAgent --model-name 01-ai/Yi-1.5-34B-Chat --domain all 

# DeepSeekAgent
python crowd.py --model-type DeepSeekAgent --model-name deepseek-ai/DeepSeek-V2-Lite-Chat --domain all
python crowd.py --model-type DeepSeekAgent --model-name deepseek-ai/DeepSeek-V2-Chat --domain all

# GemmaAgent
python crowd.py --model-type GemmaAgent --model-name google/gemma-2-2b-it --domain all
python crowd.py --model-type GemmaAgent --model-name google/gemma-2-9b-it --domain all
python crowd.py --model-type GemmaAgent --model-name google/gemma-2-27b-it --domain all

# DollyAgent
python crowd.py --model-type DollyAgent --model-name Databricks/dolly-v2-7b --domain all
python crowd.py --model-type DollyAgent --model-name Databricks/dolly-v2-12b --domain all

# FalconAgent
python crowd.py --model-type FalconAgent --model-name tiiuae/falcon-7b-instruct --domain all
python crowd.py --model-type FalconAgent --model-name tiiuae/falcon-40b-instruct --domain all   

# CohereAgent
python crowd.py --model-type CohereAgent --model-name CohereForAI/aya-expanse-8b --domain all
python crowd.py --model-type CohereAgent --model-name CohereForAI/aya-expanse-32b --domain all
python crowd.py --model-type CohereAgent --model-name CohereForAI/aya-23-8B --domain all
python crowd.py --model-type CohereAgent --model-name CohereForAI/aya-23-35B --domain all

# GraniteAgent
python crowd.py --model-type GraniteAgent --model-name ibm-granite/granite-3.0-2b-instruct --domain all
python crowd.py --model-type GraniteAgent --model-name ibm-granite/granite-3.0-8b-instruct --domain all

# PhiAgent
python crowd.py --model-type PhiAgent --model-name microsoft/phi-3.5-mini-instruct --domain all
python crowd.py --model-type PhiAgent --model-name microsoft/Phi-3-mini-4k-instruct --domain all
python crowd.py --model-type PhiAgent --model-name microsoft/Phi-3-small-8k-instruct --domain all
python crowd.py --model-type PhiAgent --model-name microsoft/Phi-3-medium-4k-instruct --domain all

# SarvamAgent
python crowd.py --model-type SarvamAgent --model-name sarvamai/sarvam-1 --domain all
python crowd.py --model-type SarvamAgent --model-name sarvamai/sarvam-2b-v0.5 --domain all