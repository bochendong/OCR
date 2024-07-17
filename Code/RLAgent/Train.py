import json
import logging

def train_rl_agent(agent, env, train_loader, action_length, path, epoches=5):
    for policy in range(epoches):
        logging.info(f'Policy: {policy}')
        total_step = 0
        batch_num = 0
        print(f'Agent with epsilon = {agent.epsilon}')
        for batch in train_loader:
            batch_num += 1
            state = env.reset(batch)

            batch_loss_sum = 0
            while (state["step"] < 512 - action_length):
                actions, q_values = agent.action(policy, state)

                new_state, reward, ocr_loss = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss_sum += loss
                total_step += action_length

            logging.info(f'Batch avg RL Loss: {batch_loss_sum / batch_num }')

    return agent