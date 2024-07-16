import json
import logging

def train_rl_agent(agent, env, train_loader, action_length, path, epoches=5):
    RL_running_sum = 0
    batch_num = 0
    for policy in range(epoches):
        logging.info(f'Policy: {policy}')
        total_step = 0

        print(f'Agent with epslion = {agent.epslion}')
        for batch in train_loader:
            batch_num += 1
            batch_loss_sum = 0
            state = env.reset(batch)

            while (state["step"] < 512 - action_length):
                actions, q_values = agent.action(policy, state)

                new_state, reward, ocr_loss = env.update(actions)

                loss = agent.learn(new_state, reward, q_values, actions)

                state = new_state

                batch_loss_sum += loss
                total_step += action_length

            RL_running_sum += batch_loss_sum
            print("Org Seq: ", batch['input_ids'])
            print("Enhanced Seq", state["selected_input_ids"])

            logging.info(f'Running Mean RL Loss: {RL_running_sum / batch_num }')


    return agent