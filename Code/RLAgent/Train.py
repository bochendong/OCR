import json
import logging

def train_rl_agent(agent, env, train_loader, action_length, path, epoches=5):
    for policy in range(epoches):
        logging.info(f'Policy: {policy}')

        RL_loss_history = {}
        OCR_loss_history = {}
        total_step = 0
        batch_num = 0

        for batch in train_loader:
            batch_num += 1
            batch_loss_sum, ocr_loss_sum = 0, 0
            state = env.reset(batch)

            while (state["step"] < 512 - action_length):
                actions, q_values = agent.action(policy, state)

                new_state, reward, ocr_loss = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss_sum += loss
                ocr_loss_sum += ocr_loss
                total_step += action_length

            RL_loss_history[total_step] = batch_loss_sum
            OCR_loss_history[total_step] = ocr_loss_sum
            logging.info(f'Batch Num: {batch_num}, Agent Loss: {batch_loss_sum/batch_num}, OCR_loss: {ocr_loss_sum/batch_num}')

        with open(path + f'/RL_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(RL_loss_history, f)

        with open(path + f'/OCR_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(OCR_loss_history, f)

    return agent