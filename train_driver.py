import os.path
import sys
import getopt
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

from Agent import Agent
from AIGame import SnakeGameAI

plt.ion()


def plot(scores, mean_scores):
    """
    helper function to plot the training results
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("# of games")
    plt.ylabel("score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def train(file_name, steps, img_name, alg):
    """
    trains the agent
    """
    # keep track of progress
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # agent/environment
    agent = Agent(alg=alg)
    environment = SnakeGameAI(w=8, h=8, agent_type=3)

    # train loop
    while agent.n_games <= steps:
        # step
        state = agent.get_state(game=environment)
        action = agent.get_action(state)
        reward, game_over, score = environment.play_step(action)
        state_new = agent.get_state(environment)
        action_new = agent.get_action(state_new)

        # train
        agent.train_sm(state, action, reward, state_new, action_new, game_over)

        # if tha game is lost: train long memory, reset game, plot partial results
        if game_over:
            # train and reset environment
            environment.reset()
            agent.n_games += 1

            # save scores
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)

            if score > record:
                record = score

            # plot
            print("Game: {}, Score: {}, Record: {}".format(agent.n_games, score, record))
            plot(plot_scores, plot_mean_scores)

    # save model
    agent.model.save(file_name)

    # save scores
    plot_scores = np.array(plot_scores)
    plot_mean_scores = np.array(plot_mean_scores)
    np.savez(file_name[:-4], arr1=plot_scores, arr2=plot_mean_scores)

    # save figure
    plt.savefig(img_name)


def main(argv):
    # init variables
    file_name = "data/model.npy"
    img_name = "data/graph.png"
    alg = "SARSA"
    steps = 10000

    # get command line arguments
    opts, args = getopt.getopt(argv, "hf:s:n:a:", ["file=", "steps=", "name=", "alg="])
    for opt, arg in opts:
        # help
        if opt == "-h":
            print("train_driver.py -f <filename> -s <steps> -n <imgname> -a <algorithm>")
            print("\t-f path to output file")
            print("\t-s steps to perform")
            print("\t-n name of the image to save")
            print("\t-a algorithm for td-error")
            print("saved models/image are put in the data directory, extension is automatically added")
            sys.exit()

        # output file
        elif opt in ("-f", "--file"):
            file_name = "data/"
            file_name += arg
            # avoid overwriting
            if os.path.exists(file_name + ".npy"):
                counter = 2
                while os.path.exists(file_name + str(counter) + ".npy"):
                    counter += 1
                file_name += str(counter) + ".npy"
            else:
                file_name += ".npy"

        # steps
        elif opt in ("-s", "--steps"):
            steps = int(arg)

        # image name
        elif opt in ("-n", "--name"):
            img_name = "data/"
            img_name += arg
            # avoid overwriting
            if os.path.exists(img_name + ".png"):
                counter = 2
                while os.path.exists(img_name + str(counter) + ".png"):
                    counter += 1
                img_name += str(counter) + ".png"
            else:
                img_name += ".png"

        # algorithm
        elif opt in ("-a", "--alg"):
            alg = arg

    # execute
    train(file_name=file_name, steps=steps, img_name=img_name, alg=alg)


if __name__ == "__main__":
    main(sys.argv[1:])
