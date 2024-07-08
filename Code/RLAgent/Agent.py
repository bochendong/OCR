import math
import logging
import numpy as np

import torch
import torch.optim as optim

from .Model import RLAgent

class Q_LearningAgent(object):
    def __init__(self, action_length, device, criterion, 
                learning_rate=1e-6, gamma = 0.99, slow_start = True):
        self.gamma = gamma
        self.action_length = action_length
        self.slow_start = slow_start
        self.device = device
        self.epsilon_schedule = [1.0, 0.9, 0.8, 0.7, 0.6]

        self.Q_net = RLAgent(device).to(device)
        self.optimizer = optim.Adam(self.Q_net .parameters(), lr=learning_rate)
        self.criterion = criterion

    def predict(self, state, step):
        return self.Q_net(state, step // self.action_length).squeeze(0)
    
    def action(self, epoch, state):
        step = state["step"]
        q_values = self.predict(state, step)
        action_type = self.policy(epoch, step)

        if (action_type == "Greedy"):
            actions = list(range(step, step + self.action_length))
        else:
            actions = torch.topk(q_values, self.action_length, dim=-1).indices.squeeze().tolist()

        return actions, q_values
    
    def policy(self, epoch, step):
        epsilon = self.epsilon_schedule[epoch]
        if (self.slow_start and (np.random.rand() < epsilon or step < 35)):
            action_type = "Greedy"
        else:
            action_type = "Q_NET_PREDICT"

        return action_type

    def learn(self, state, reward, q_values, actions):
        step = state["step"]
        with torch.no_grad():
            next_q_values = self.predict(state, step)
            max_next_q_value = torch.max(next_q_values).item()
            target_q_value = reward + self.gamma * max_next_q_value
        
        target_q_values = torch.tensor([target_q_value] * self.action_length).to(self.device)
        selected_q_values = q_values[actions]

        loss = self.criterion(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()







