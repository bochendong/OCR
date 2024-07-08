import json
import logging
import matplotlib.pyplot as plt

def SetupLogging(file_name):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(file_name), logging.StreamHandler()],
                        force=True)
    

def PlotAgentLoss(file_path, policy):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    OCR_json_path = file_path + f'/OCR_loss_history_policy_{policy}.json'
    with open(OCR_json_path, 'r') as f:
        loaded_OCR_loss_history = json.load(f)
        
    OCR_iterations = list(map(float, loaded_OCR_loss_history.keys()))
    OCR_losses = list(map(float, loaded_OCR_loss_history.values()))
        
    axs[0].plot(OCR_iterations, OCR_losses, label=f'OCR Policy {policy}')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'OCR Policy {policy} Training Loss History')
    axs[0].legend()
    axs[0].grid(True)

    # Plot RL Loss History
    RL_json_path = file_path + f'/RL_loss_history_policy_{policy}.json'
    with open(RL_json_path, 'r') as f:
        loaded_RL_loss_history = json.load(f)
        
    RL_iterations = list(map(float, loaded_RL_loss_history.keys()))
    RL_losses = list(map(float, loaded_RL_loss_history.values()))
        
    axs[1].plot(RL_iterations, RL_losses, label=f'RL Policy {policy}')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'RL Policy {policy} Training Loss History')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(file_path + f'/loss_history_policy_{policy}.png')