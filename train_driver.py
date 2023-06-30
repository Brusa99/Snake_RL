import matplotlib.pyplot as plt
from IPython import display

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


def train():
    """
    trains the agent
    """
    # keep track of progress
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # agent/environment
    agent = Agent()
    environment = SnakeGameAI(w=8, h=8)

    # train loop
    while True:
        state_old = agent.get_state(game=environment)

        # move
        action = agent.get_action(state_old)
        reward, game_over, score = environment.play_step(action)
        state_new = agent.get_state(environment)

        # train
        agent.train_sm(state_old, action, reward, state_new, game_over)
        agent.remember(state_old, action, reward, state_new, game_over)

        # if tha game is lost: train long memory, reset game, plot partial results
        if game_over:
            # train and reset environment
            environment.reset()
            agent.n_games += 1
            agent.train_lm()
            
			# save scores
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)

            if score > record:
                record = score
                agent.model.save("data/model.pth")

            # plot
            print("Game: {}, Score: {}, Record: {}".format(agent.n_games, score, record))
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
