import argparse
import new_agent
import environment
import runner
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='test bed for dynamic programming algorithms')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

parser_RD = subparsers.add_parser(
    'RD', description='Random Agent')
parser_VI = subparsers.add_parser(
    'VI', description='Value Iteration agent')
parser_PI = subparsers.add_parser(
    'PI', description='Policy Iteration agent')
parser_QL = subparsers.add_parser(
    'QL', description='Q-Learning agent')
parser_SARSA = subparsers.add_parser(
    'SARSA', description='SARSA agent')
parser_ALL = subparsers.add_parser(
    'ALL', description='Run RD, QL and SARSA agent (if implemnted) and plot performance')

parsers = [parser_RD, parser_VI, parser_PI, parser_QL, parser_SARSA, parser_ALL]

arg_dico = {'RD': new_agent.Agent,
            'VI': new_agent.ValueIteration,
            'PI': new_agent.PolicyIteration,
            'QL': new_agent.QLearning,
            'SARSA': new_agent.SARSA
            }

def plot_results(sum_of_rewards, list_legends):
    for sum_rew, legend in zip(sum_of_rewards, list_legends):
        plt.plot(sum_rew, label=legend)
    plt.legend(loc='lower right')
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards')
    plt.show()

def plot_policy(policy, V, name):
    action = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    for i in range(4):
        if i != 0:
            plt.figure(i)
        policy_ = policy[:, i].reshape((4, 4))
        policy_[0, 0] = -1
        policy_[3, 3] = -1
        plt.matshow(policy_)
        plt.title('Algo: ' + name + ' - Policy : ' + action[i])
    plt.show()

def run_agent(nb_episodes, args):
    env_class = environment.EnvironmentGridWorld()
    agent_class = arg_dico[args.agent]

    print("Running a single instance simulation...")
    name = args.agent
    my_runner = runner.Runner(env_class, agent_class(env_class), name)
    if name in ["RD", "QL", "SARSA"]:
        final_reward = my_runner.loop(nb_episodes)
        plot_results([final_reward], [args.agent])
    elif name in ["PI", "VI"]:
        policy, V = my_runner.loop(nb_episodes)
        plot_policy(policy, V, name)

def main():
    nb_episodes = 500
    args = parser.parse_args()
    if args.agent != "ALL":
        run_agent(nb_episodes, args)
    else:
        list_final_reward = []
        list_agent = []
        for agent in ["RD", "QL", "SARSA"]:
            env_class = environment.EnvironmentGridWorld()
            agent_class = arg_dico[agent]

            print("Running a single instance simulation...")
            my_runner = runner.Runner(env_class, agent_class(env_class), agent)
            final_reward = my_runner.loop(nb_episodes)
            list_final_reward.append(final_reward)
            list_agent.append(agent)

        plot_results(list_final_reward, list_agent)

if __name__ == "__main__":
    main()
