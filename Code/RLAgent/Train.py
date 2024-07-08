import json
import logging

def train_rl_agent(agent, env, train_loader, action_length, path, policys=5):
    for policy in range(policys):
        logging.info(f'Policy: {policy}')

        loss_history = {}
        total_step = 0

        for batch_num, batch in enumerate(train_loader):
            batch_loss = 0
            state = env.reset(batch)
            while (state["step"] < 512 - action_length):
                reward = 0
                actions, q_values = agent.action(policy, state)

                new_state, reward = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss += loss
                total_step += action_length

                loss_history[total_step] = loss
            
            logging.info(f'Batch Num: {batch_num}, Batch Loss: {batch_loss}')
        
        with open(path + f'/loss_history_policy_{policy}.json', 'w') as f:
            json.dump(loss_history, f)

    return agent