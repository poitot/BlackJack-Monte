import sys
import classes
import numpy as np
import gym
from blackjack_env2.env.blackjack_env2 import BlackjackEnv
import datetime
import matplotlib.pyplot as plt

def main():
    env = gym.make('blackjack-v2')
    count_range = np.arange(-13, 14)


    states = [(x, y, z) for x in range(4,22) for y in range(1,11) for z in [True,False]] 

    state_counts = {s:0 for s in states}

    #suit counts [clubs, diamonds, hearts, spades]
    counts = [0, 0, 0, 0]

    runs = 10000000

    for i in range(runs):
        if i % 100000 == 0:
            print (i)
        env.reset()

        cards = env.dealerHand.cards + env.playerHand.cards
        state_counts[env._get_obs()] += 1

    #print(state)

    state_probs = {k: v / runs for k, v in state_counts.items()}
    write = open("main_probs{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")), 'w')
    write.write(str(state_probs))
    write.close()

    write = open("main_count{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")), 'w')
    write.write(str(state_counts))
    write.close()
if __name__ == "__main__":
    sys.exit(main())