import json
import matplotlib.pyplot as plt

def PlotAgentLoss(file_path):
    plt.figure(figsize=(10, 5))

    for policy in range(5):
        json_path = file_path + f'/loss_history_policy_{policy}.json'

        with open(json_path, 'r') as f:
            loaded_loss_history = json.load(f)
        
        plt.plot(loaded_loss_history, label=f'Policy {policy}')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(file_path + '/training_loss_history.png')
    plt.show()