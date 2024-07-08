import os
import torch
import torch.nn as nn


from Code.DataSet.DataPreprocess import DataPreprocessor
from Code.RLAgent.Agent import Q_LearningAgent
from Code.RLAgent.Train import train_rl_agent
from Code.Environment.Environment import Env
from Code.Utils.GetBaseModel import getBaseModel
from Code.Utils.Performance import compute_metrics
from Code.Utils.Logging import setup_logging

if __name__ == "__main__":
    learning_rate=1e-6 
    gamma=0.99
    action_length = 32
    reward = "normal"
    slow_start = True

    Log_path = f'./Log/action_length={action_length}_reward={reward}_slow_start={slow_start}'

    if (os.path.exists(Log_path) == False):
        os.mkdir(Log_path)

    setup_logging(Log_path + "/log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    funsd = DataPreprocessor()

    train_loader, test_loader = funsd.get_data_loader()

    model = getBaseModel(funsd.id2label, funsd.label2id).to(device)
    env = Env(model, device)


    criterion = nn.SmoothL1Loss()
    agent = Q_LearningAgent(action_length, device, criterion, 
                            learning_rate=learning_rate, gamma = gamma, slow_start = slow_start)

    train_rl_agent(agent, env, train_loader, action_length, path = Log_path)

    


