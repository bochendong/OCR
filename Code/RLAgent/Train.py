import json
import logging

def train_rl_agent(agent, env, train_loader, action_length, path, policys=5):
    for policy in range(policys):
        logging.info(f'Policy: {policy}')

        RL_loss_history = {}
        OCR_loss_history = {}
        total_step = 0

        for batch_num, batch in enumerate(train_loader):
            batch_loss = 0
            state = env.reset(batch)
            while (state["step"] < 512 - action_length):
                reward = 0
                actions, q_values = agent.action(policy, state)

                new_state, reward, ocr_loss = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss += loss
                total_step += action_length

                RL_loss_history[total_step] = loss
                OCR_loss_history[total_step] = ocr_loss
            
            logging.info(f'Batch Num: {batch_num}, Batch Loss: {batch_loss/(batch_num + 1)}, OCR_loss: {ocr_loss/(batch_num + 1)}')
        
        with open(path + f'/RL_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(RL_loss_history, f)

        with open(path + f'/OCR_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(OCR_loss_history, f)

        

    return agent