import gym
from gym import spaces
from gym.utils import seeding
import importlib
from classes import Hand, Deck, Card
import numpy as np


"""
21 + 3 side bet enviroment
"""



class SideEnv(gym.Env):

    def cmp(self, a, b):
        return float(a > b) - float(a < b)
    

    def draw_card(self):
        card = self.deck.get_card()
        return card

    def draw_card(self, card):
        card = self.deck.deck_cards.remove(card)
        return card


    def draw_hand(self, np_random):
        h = Hand()
        h.cards = [self.deck.get_card(), self.deck.get_card()]
        h.usableAce = self.usable_ace(h)
        return h

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Tuple((
            #hi lo count
            spaces.Discrete(32),

            # player card 1 val (1 - 10) & suit (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)
            spaces.Discrete(10),
            spaces.Discrete(4),
            # player card 2 val (1 - 10) & suit (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)
            spaces.Discrete(10),
            spaces.Discrete(4),
            # dealer card val (1 - 10) & suit (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)
            spaces.Discrete(10),
            spaces.Discrete(4)))
            

            #suit counts
            #spaces.Discrete(15),
            #spaces.Discrete(15),
            #spaces.Discrete(15),
            #spaces.Discrete(15)
        self.seed()

    def check_side(self, cards , side_bet, deck):
        count_max = 14
        true_count = deck.get_true_suit_counts()
        reward = 0
        # check if hand is suited
        if (cards[0].suit == cards[1].suit == cards[2].suit):
            suited = True
        else:
            suited = False

        # check for three of a kind : pays 30/1
        if (cards[0].name == cards[1].name == cards[2].name):

            # check for suited three of a kind : pays 100/1
            if (suited):
                reward = 9

            else:
                reward = 9

        # sorts cards in ascending order to check for straight : e.g. 5-6-7, 2-3-4 : Aces are counted as 1 or 11
        card_vals = []
        card_vals.append(cards[0].value)
        card_vals.append(cards[1].value)
        card_vals.append(cards[2].value)
        card_vals = sorted(card_vals)
        # check for straight by comparing card values, result is squared to account for negative returns
        # if ((9 - 8)**2 == 1 and (8 - 7)**2 == 1) = true
        # if (())
        if ((card_vals[2] - card_vals[1])**2 == 1 and (card_vals[1] - card_vals[0])**2 == 1):
            # straight flush : straight with matching suits  : pays 40/1
            if (suited):
                reward = 9
            # straight : non-suited straight  : pays 10/1
            else:
                reward = 9

        # flush : all cards suits match : pays 5/1
        if (reward == 0 and suited):
            reward = 9

        if reward > 0:
            chips = (reward * side_bet) + side_bet
        else:
            chips = (-1 * side_bet)
        cards_str = ""

        # negatively rewards agent if no bet if making a bet would have resulted in a positive reward
        #if (side_bet == 0 and reward > 0):
            #reward = -1 * reward
        
        if side_bet > 0 and reward == 0:
            reward = -1 * side_bet

        return reward, chips

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        #np.random.seed(seed)
        return [seed]

    def step(self, action, cards, deck):
        assert self.action_space.contains(action)
        reward = 0.0
        chips = 0

        
        if action == 0:
            side_bet = 0
            reward, chips = self.check_side(cards, side_bet, deck)
            
        if action == 1:
            side_bet = 1
            reward, chips = self.check_side(cards, side_bet, deck)
        
        if action == 2:
            side_bet = 5
            reward, chips = self.check_side(cards, side_bet, deck)

        if action == 3:
            side_bet = 10
            reward, chips = self.check_side(cards, side_bet, deck)

        if action == 4:
            side_bet = 25
            reward, chips = self.check_side(cards, side_bet, deck)

        if action == 5:
            side_bet = 50
            reward, chips = self.check_side(cards, side_bet, deck)

        if action == 6:
            side_bet = 100
            reward, chips = self.check_side(cards, side_bet, deck)

        
        return self._get_obs(deck), reward, True, chips, deck.get_true_suit_counts()

    # reward is discounted based on the true count of suits
    def get_discount(self, deck, reward):
        if not(reward == 0):
            count = deck.get_true_suit_counts()
            discount = reward / (count )
        else:
            return reward

    def _get_obs(self, deck):
        #a = tuple(deck.get_true_suit_counts())
        #a = tuple(deck.count_suits)
        count = int(deck.get_true_suit_counts())
        return (deck.count_suits[0]//deck.num_decks, deck.count_suits[1]//deck.num_decks)

    def get_hands(self):
        return self.playerHand, self.dealerHand
        
    def reset(self, env):
        self.dealerHand = env.dealerHand

        self.playerHand = env.playerHand
            
        return self._get_obs(env.deck)

    def generate_episode(self, state, env):

        #print("looking for state: ", state, "ebv state: ", env.deck.count_suits[0])

        env.deck_by_count(state)




