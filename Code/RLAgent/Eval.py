from ..Utils.Performance import compute_metrics

def EvalRlAgent(agent, env, test_loader, action_length, policy):
    average_test_f1 = []
    for batch in test_loader:
        state = env.reset(batch)
        while (state["step"] < 512 - action_length):
            actions, _ = agent.action(policy, state)
            new_state, _, _ = env.update(actions)
            state = new_state
        outputs = env.get_result(state)
        metrics = compute_metrics(outputs, state["selected_target"])
        f1 = metrics['f1']
        average_test_f1.append(f1)

    average_test_f1 = sum(average_test_f1) / len(average_test_f1)
    return average_test_f1
    