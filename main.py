import json
import sys
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from src.causal_explanations import reccover, eval_policy
from envs.gridworld_traffic_env import GridworldTrafficEnv, SymbolicFlatObsWrapper
from envs.taxi import TaxiEnv
from algs.parametrized.dqn_agent_parametrized import DQNAgentParametrized
from algs.simple.dqn_agent import DQNAgent
from src.model_util import seed_everything


def main(task, train):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device = {}'.format(device))

    params_file = 'settings/{}.json'.format(task)

    # Read settings file
    with open(params_file, 'r') as f:
        params = json.loads(f.read())
        print(params)

    if task == 'taxi':
        env = TaxiEnv(max_timesteps=200)
    elif task == 'gridworld':
        env = GridworldTrafficEnv()
        env = SymbolicFlatObsWrapper(env)

    env_name = env.name
    check_env(env)

    true_policy_path = 'trained_models/{}/simple/policy_correct.pt'.format(task)
    conf_policy_path = 'trained_models/{}/simple/policy_confused.pt'.format(task)
    graph_policy_path = 'trained_models/{}/graph_policy_all_masks.pt'.format(task)
    results_graph_policy = 'results/{}/results.csv'.format(task)

    # Initialize policies
    masks =  torch.from_numpy(np.array(params['masks']))
    policy_confused = DQNAgent(env.state_dim, params['hidden_size'], env.num_actions, env.state_dim, device)
    policy_correct = DQNAgent(env.state_dim, params['hidden_size'], env.num_actions, env.state_dim, device)
    graph_policy = DQNAgentParametrized(2 * env.state_dim, 512, env.num_actions, env.state_dim, device, masks)

    if train:
        env.set_confused(True)
        policy_confused.train_dqn(env, conf_policy_path, num_episodes=3000)
        env.set_confused(False)
        policy_correct.train_dqn(env, true_policy_path, num_episodes=3000)
        env.set_confused(None)
        graph_policy.train_dqn(env, graph_policy_path, eval_path=results_graph_policy)
    else:
        policy_confused.load(from_cuda=True, path=conf_policy_path)
        policy_correct.load(from_cuda=True, path=true_policy_path)
        graph_policy.load(graph_policy_path)

    env.set_confused(None)
    eval_path = 'results/{}'.format(task) + '/eval_{}_{}_{}.csv'

    # Evaluate policies
    total_correct = eval_policy(policy_correct, env)
    total_confused = eval_policy(policy_confused, env)

    print('Confused policy: {}'.formawweqwezxt(total_confused))
    print('Correct policy: {}'.format(total_correct))

    # Evaluate ReCCoVER on the confused policy
    print('------------------- EVALUATING CONFUSED POLICY -------------------')
    reccover(policy_confused,
             graph_policy,
             env,
             params['features'],
             params['feature_values'],
             eval_path,
             env_name,
             params['confused_mask'],
             params['k'])

    # Evaluate ReCCoVER on the correct policy
    print('------------------- EVALUATING CORRECT POLICY -------------------')
    reccover(policy_correct,
             graph_policy,
             env,
             params['features'],
             params['feature_values'],
             eval_path,
             env_name,
             params['true_mask'],
             params['k'])


if __name__ == '__main__':
    args = sys.argv
    try:
        task_index = args.index('--task')
        task = sys.argv[task_index + 1]
    except ValueError:
        print("Provide type of task")

    train = False
    if '--train' in args:
        train = True

    print('Task = {}'.format(task))
    print('Train = {}'.format(train))
    main(task, train)