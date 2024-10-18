# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-10-18

import os
import torch
os.environ['HF_HUB_CACHE'] = '/mnt/data'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from transformers import pipeline
from transformers import BitsAndBytesConfig

class AbstractAgent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
            device_map="auto",
        )     
        
    def generate(self, messages, max_new_tokens=256, temperature=0.0):
        if temperature > 0.0:
            response = self.pipe(messages, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        else:
            response = self.pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        last_response = response[0]["generated_text"][-1]
        return last_response
    
    
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Respond using Json."},
        {"role": "user", "content": "Hello. Write me a poem."},
    ]
    
    agent = AbstractAgent("meta-llama/Llama-3.1-70B-Instruct")
    
    response = agent.generate(messages, max_new_tokens=256, temperature=0.0)
    print(response)


