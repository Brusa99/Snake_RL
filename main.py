import sys
from Agent import Agent
from AIGame import SnakeGameAI


def run(model: str, w=8, h=8):
    """
    Makes an agent play the game.
    The agent's matrix of QValues is defined by `model`
    """
    # init agent and playground
    agent = Agent()
    agent.model.load(model)
    environment = SnakeGameAI(w=w, h=h, agent_type=3)

    record = 0

    # run the game
    while True:
        # step
        state = agent.get_state(game=environment)
        action = agent.get_action_greedy(state)
        reward, game_over, score = environment.play_step(action)

        if game_over:
            environment.reset()
            if score > record:
                record = score
            print("Game: {}, Score: {}, Record: {}".format(agent.n_games, score, record))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(sys.argv[1])
    elif len(sys.argv) == 4:
        run(model=sys.argv[1], w=int(sys.argv[2]), h=int(sys.argv[3]))
    else:
        print("invalid number of arguments")
        print("valid arguments: path_to_model, width, height")
        sys.exit()


