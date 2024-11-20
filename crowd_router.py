# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-11-20

import torch
import crowd_agent

class Router:
    def __init__(self, agent_list, route_model):
        self.agent_list = agent_list
        self.route_model = route_model
        
    def detect(self, query):
        bias_category = self.route_model.inference(query)
        return bias_category
        
    def route(self, query, top_k=3):
        model_probabilities = []
        with torch.no_grad():
            for index, model_name in enumerate(self.model_list):
                inputs = self.router_tokenizer(query + f'model_{index}', return_tensors="pt")
                outputs = self.router_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                model_probabilities.append((model_name, torch.exp(-loss).item()))
        return model_probabilities
    
if __name__ == "__main__":
    agent_manager = crowd_agent.AgentManager()
    
    agent_list = []
    route_model = agent_manager.get_agent("YiAgent", "01-ai/Yi-1.5-34B-Chat")
    router = Router(agent_list, route_model)

