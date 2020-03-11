import argparse
import sys
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
import datetime
import os



import gym
from blackjack_env2.env.blackjack_env2 import BlackjackEnv
from gym import wrappers, logger


# Rewards in Blackjack happen at the end of the game, no discounting
GAMMA = 1.0

logger = logging.getLogger()


class BlackjackAgent(object):
    def __init__(self, action_space, epsilon_decay):
        self.action_space = action_space
        self.action_space_side = [0, 1]
        self.nA = action_space.n
        self.epsilon_decay = epsilon_decay

        # initialize Q value and N count dictionaries
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.N = defaultdict(lambda: np.zeros(action_space.n))

        # policy and action-values for splitting pairs
        self.Q_split = defaultdict(lambda: np.zeros(2))
        self.N_split = defaultdict(lambda: np.zeros(2))
        
        # initialize side bet value and count dicts
        self.Q_side = defaultdict(lambda: np.zeros(1))
        self.N_side = defaultdict(lambda: np.zeros(1))

        # track episode num for epsilon
        self.i_episode = 0

    def log_observation(self, observation):
        player_hand, dealer_showing, usable_ace = zip(observation)
        logger.debug('player hand:{}, dealer showing:{}, usable ace:{}'.format(player_hand[0], dealer_showing[0], usable_ace[0]))

    def log_done(self, observation, reward):
        self.log_observation(observation)
        logger.debug('final reward:{}\n'.format(reward))

    def get_policy_for_observation(self, Q_s, epsilon):
        """ calculates the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
        return policy_s



    def choose_action(self, observation, epsilon):
        """ if observation in Q dict, choose random action with probability of eps otherwise sample from action space """
            
        obs = (observation[0], observation[1], observation[2])
        pol = self.Q

        if observation in self.Q:
            a = np.random.choice(np.arange(self.nA))
            p = self.get_policy_for_observation(self.Q[observation], epsilon)

            action = np.argmax(p)
    

        else:
            action = self.action_space.sample()

        return action

    def choose_side_bet(self, observation, epsilon):
        if observation in self.Q_side:
            policy_s = np.ones(self.nA) * epsilon / self.nA
            best_a = np.argmax(Q_side[observation])
            policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
            return policy_s

        else:
            return 1

    def update_action_val_function(self, episode):
        """ updates the action-value function Q and N count dictionaries for every observation in one episode """
        observations, actions, rewards = zip(*episode)
        discounts = np.array([GAMMA**i for i in range(len(rewards)+1)])
        for i, observation in enumerate(observations):
            old_Q = self.Q[observation][actions[i]]
            old_N = self.N[observation][actions[i]]
            self.Q[observation][actions[i]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
            self.N[observation][actions[i]] += 1

    def generate_episode(self, env):
        """ start new episode and update action-value function until episode terminates """
        self.i_episode += 1
        # decay epsilon
        epsilon = 1.0/((self.i_episode/self.epsilon_decay) + 1)
        episode = []
        observation = env.reset()
        self.log_observation(observation)
        while True:
            action = self.choose_action(observation, epsilon)
            logger.debug('HIT' if action else 'STICK')
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                self.log_done(next_observation, reward)
                break
            else:
                self.log_observation(next_observation)
            observation = next_observation
        return episode

    def generate_episode(self, env, side_bet):
        """ start new episode and update action-value function until episode terminates """
        self.i_episode += 1
        # decay epsilon
        epsilon = 1.0/((self.i_episode/self.epsilon_decay) + 1)
        episode = []
        observation = env.reset()
        
        if side_bet > 0:
            side_bet_winnings = 0
            player_hand, dealer_hand = zip(env.get_hands())
            cards = [player_hand[0].cards[0], player_hand[0].cards[1], dealer_hand[0].cards[0]]
            """
            21 + 3:

            change observation to contain player & dealer hands in tuple: [player_hand[card_1, card_2], dealer_card[1]]

            check for poker hands: flush, straight, three of a kind, etc...

            odds source : https://www.telegraph.co.uk/betting/casino-guides/blackjack/21-plus-3/

            
            """

            # check if hand is suited
            if (cards[0].suit == cards[1].suit == cards[0].suit):
                suited = True
            else:
                suited = False

            # check for three of a kind : pays 30/1
            if (cards[0].value == cards[1].value == cards[0].value):

                # check for suited three of a kind : pays 100/1
                if (suited):
                    side_bet_winnings = side_bet * 100

                else:
                    side_bet_winnings = side_bet * 30

            # sorts cards in ascending order to check for straight : e.g. 5-6-7, 2-3-4 : Aces are counted as 1 or 11
            card_vals = player_hand[0].getValues()
            card_vals.append(dealer_hand[0].cards[0].value)
            cards = sorted(card_vals)
            # check for straight by comparing card values, result is squared to account for negative returns
            # if ((9 - 8)**2 == 1 and (8 - 7)**2 == 1) = true
            # if (())
            if ((cards[2] - cards[1])**2 == 1 and (cards[1] - cards[0])**2 == 1):
                # straight flush : straight with matching suits  : pays 40/1
                if (suited):
                    side_bet_winnings = side_bet * 40
                # straight : non-suited straight  : pays 10/1
                else:
                    side_bet_winnings = side_bet * 10

            # flush : all cards suits match : pays 5/1
            if (side_bet_winnings == 0 and suited):
                side_bet_winnings = side_bet * 5


        self.log_observation(observation)
        while True:
            action = self.choose_action(observation, epsilon)
            logger.debug('HIT' if action else 'STICK')
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                self.log_done(next_observation, reward)
                break
            else:
                self.log_observation(next_observation)
            observation = next_observation
        return episode

def learn(base_dir='honours 3', num_episodes=100000, epsilon_decay=8000):
    env = gym.make('blackjack-v2')
    env = wrappers.Monitor(env, directory=base_dir, force=True, video_callable=False)

    agent = BlackjackAgent(env.action_space, epsilon_decay)

    rewards = np.zeros(num_episodes)
    total_rewards = 0

    for i in range(num_episodes):
        if i % 1000 == 0:
            logger.debug('\rEpisode {}/{}.'.format(i, num_episodes))
            print('\rEpisode {}/{}.'.format(i, num_episodes), 1.0/((i/epsilon_decay) + 1))
        
        side_bet = agent.choose_side_bet(env.deck, epsilon_decay)
        episode = agent.generate_episode(env, side_bet)

        agent.update_action_val_function(episode)
        X = episode
        total_rewards += episode[0][2]
        rewards[i] = total_rewards 

    # obtain the policy from the action-value function
    # e.g. generate  ((4, 7, False), 1)   HIT      ((18, 6, False), 0)  STICK
    policy = dict((k, np.argmax(v)) for k, v in agent.Q.items())

    env.close()


    return policy, agent.Q

def choose_action_by_policy(action_space, policy, observation):
    """ selects action based on trained policy """
    if observation in policy:
        action = policy[observation]
    else:
        # observation not found in policy, usually if there hasn't been enough training episodes
        action = action_space.sample()
    return action

def score(policy, num_episodes):
    """ average score using policy after num_episodes """
    env = gym.make('blackjack-v2')
    rewards = []
    rewards_sum = 0
    num_episodes = num_episodes // 10
    for _ in range(num_episodes):
        observation = env.reset()
        
        while True:
            action = choose_action_by_policy(env.action_space, policy, observation)
            next_observation, reward, done, _ = env.step(action)
            rewards_sum += reward
            if done:
                rewards.append(rewards_sum)
                break
            observation = next_observation
    env.close()

    return rewards_sum / num_episodes

def plot_policy(policy, plot_filename="plot.png"):

    def get_Z(player_hand, dealer_showing, usable_ace):
        b = []
        counts = []

        

        for c in policy:
            b.append(([c[0], c[1], c[2]]))
            #counts.append(c[3])
            # ([4, 2, False], 0)

        d = (player_hand, dealer_showing, usable_ace)

        if d in policy:
            return policy[player_hand, dealer_showing, usable_ace]
        else:
            return 1

        if d in b:
            count_indexes = []
            counts_vals = []
            for i, item in b:
                if item == d:
                    count_indexes.append(i)
            for indx in count_indexes:
                counts_vals.append(counts[indx]) 
            return policy[player_hand, dealer_showing, usable_ace, min(counts_vals, key=lambda x:abs(x-0))]
        else:
            return 0

    def get_figure2():
        fig2 = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
        f2_ax1 = fig2.add_subplot(spec2[0, 0])
        f2_ax2 = fig2.add_subplot(spec2[0, 1])
        f2_ax3 = fig2.add_subplot(spec2[1, 0])
        f2_ax4 = fig2.add_subplot(spec2[1, 1])



    def get_figure():
        titles = ['No Usable Ace', 'Usable Ace', 'No Usable Ace  (Double)', 'No Usable Ace (Double)']

        x_range = np.arange(1, 11)
        y_range = np.arange(11, 22)
        grid = np.meshgrid(x_range, y_range)

        fig, ax = plt.subplots(2, 2)
        X, Y = np.meshgrid(x_range, y_range)

        t = 0
        for y in range(2):
            for x in range(2):

                axis_dir = "right"


                ax[x, y].set_xlabel('Dealer Showing')
                ax[x, y].set_ylabel('Player Hand')
                ax[x, y].grid(color='black', linestyle='-', linewidth=1)

                usable_ace = x
                double = y
                Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])

                if not(double):
                    for i in range(11):
                        for j in range(10):
                            if Z[i][j] == 2:
                                Z[i][j] = 1

                num_actions =  0
                if double:
                    num_actions = 3
                else:
                    num_actions = 2
                surf = ax[x, y].imshow(Z, cmap=plt.get_cmap('Accent', 3), vmin=0, vmax=3 - 1, extent=[0.5, 10.5, 10.5, 21.5])
                plt.setp(ax, xticks=x_range, xticklabels=('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'), yticks=y_range)
                

                ax[x, y].set_title(titles[t], fontsize=16)
                t += 1


                divider = make_axes_locatable(ax[x, y])
                cax = divider.append_axes("right", size="5%", pad=0.5)
                cbar = plt.colorbar(surf, ticks=[0, 1, 2], cax=cax)
                cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)', '2 (DOUBLE)'])
                cbar.ax.invert_yaxis() 
                



        
            


    get_figure()
    plt.show()

    plt.savefig(plot_filename)

#
#   usable ace hands not being added to policy
#   check usable ace emthod is updating hand.usableace flag
#   ace start val was 11 instead of 1 so usable_ace() always returned false
#
def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--base-dir', default='blackjack-1', help='Set base dir.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')    
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)
    
    num_episodes = 100000
    epsilon_decay = 50000


    policy, Q = learn(args.base_dir, num_episodes, epsilon_decay)

    final_average_return = score(policy, num_episodes)
    os.mkdir("{}_results".format(str(datetime.datetime.now()).replace(' ', '_').replace('.', ':').replace(':', '-')) + ".txt", 755) 
    write = open("policy.txt") 
    write.write(str(policy))
    write.close()
    write = open("val_func.txt")
    write.write(str(Q))
    write.close()

    plt.plot(final_average_return)
    #plt.show()
    #logger.info("final average returns: {}".format(final_average_return))

    plot_policy(policy, "diag_{}_{}_{}.png".format(num_episodes, epsilon_decay, 0))
    



    return 0


if __name__ == "__main__":
    sys.exit(main())