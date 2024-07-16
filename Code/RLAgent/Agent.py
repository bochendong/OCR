import math
import logging
import numpy as np
import random

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
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.Q_net.train()
        self.epsilon = 1

    def predict(self, state, step):
        return self.Q_net(state, step // self.action_length).squeeze(0)
    
    def action(self, epoch, state):
        step = state["step"]
        q_values = self.predict(state, step)
        action_type = self.policy(epoch, step)

        if (action_type == "Greedy"):
            remain_input_ids = state["remain_input_ids"]
            actions = [0] * self.action_length
            counter = 0
            for i, remain_id in enumerate(remain_input_ids):
                if (remain_id != 0):
                    actions[counter] = i
                    counter += 1
                    if (counter >= self.action_length):
                        break
        else:
            Q_net_actions = torch.topk(q_values, self.action_length, dim=-1).indices.squeeze().tolist()
            actions = Q_net_actions
        
        return actions, q_values
        '''
        step = state["step"]
        q_values = self.predict(state, step)
        action_type = self.policy(epoch, step)
        Q_net_actions = torch.topk(q_values, self.action_length, dim=-1).indices.squeeze().tolist()

        if (action_type == "Greedy"):
            remain_input_ids = state["remain_input_ids"]
            actions = [0] * self.action_length
            counter = 0
            for i, remain_id in enumerate(remain_input_ids):
                if (remain_id != 0):
                    actions[counter] = i
                    counter += 1
                    if (counter >= self.action_length):
                        break
        else:
            actions = Q_net_actions'''

    
    def policy(self, epoch, step):
        if step < self.action_length:
            self.epsilon = max(self.epsilon * 0.99, 0.6)

        if (self.slow_start and (np.random.rand() < self.epsilon or step < 35)):
            action_type = "Greedy"
        else:
            action_type = "Q_NET_PREDICT"

        return action_type

    def learn(self, state, reward, q_values, actions):
        step = state["step"]
        with torch.no_grad():
            next_q_values = self.predict(state, step)
            Q_net_actions = torch.topk(next_q_values, self.action_length, dim=-1).indices.squeeze().tolist()

        selected_q_values = q_values[actions].to(self.device)
        target_q_values = reward + next_q_values[Q_net_actions].to(self.device)

        loss = self.criterion(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
