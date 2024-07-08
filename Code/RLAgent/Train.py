import json
import logging
from ..Utils.Performance import compute_metrics

def train_rl_agent(agent, env, train_loader, action_length, path, policys=5):
    labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
    id2label = {v: k for v, k in enumerate(labels)}
    for policy in range(policys):
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
                reward = 0
                actions, q_values = agent.action(policy, state)

                new_state, reward, ocr_loss = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss_sum += loss
                ocr_loss_sum += ocr_loss
                total_step += action_length

            RL_loss_history[total_step] = batch_loss_sum/(batch_num)
            OCR_loss_history[total_step] = ocr_loss_sum/(batch_num)
            logging.info(f'Batch Num: {batch_num}, Batch Loss: {batch_loss_sum/(batch_num)}, OCR_loss: {ocr_loss_sum/(batch_num)}')
        
        outputs = env.get_result(state)
        metrics = compute_metrics(outputs, state["selected_target"], id2label)
        logging.info(f"Batch {batch_num}, Accuracy: {metrics['accuracy'],} F1: {metrics['f1']}")

        with open(path + f'/RL_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(RL_loss_history, f)

        with open(path + f'/OCR_loss_history_policy_{policy}.json', 'w') as f:
            json.dump(OCR_loss_history, f)

    return agent