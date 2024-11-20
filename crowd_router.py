# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-11-20

import crowd_agent

class Router:
    def __init__(self, agent_list, route_model):
        self.agent_list = agent_list
        self.route_model = route_model
        
    def detect(self, query):
        bias_category = self.route_model.inference(query)
        return bias_category
        
    def route(self, query, top_k=3):
        pass
    
if __name__ == "__main__":
    agent_manager = crowd_agent.AgentManager()
    
    agent_list = []
    route_model = agent_manager.get_agent("YiAgent", "01-ai/Yi-1.5-34B-Chat")
    router = Router(agent_list, route_model)

