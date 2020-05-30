import gym
from gym import spaces
from gym.utils import seeding
import importlib
from classes import Hand, Deck, Card
import numpy as np






class BlackjackEnv(gym.Env):

    def cmp(self, a, b):
        return float(a > b) - float(a < b)

    def draw_card_(self, card):
        c = self.deck.get_card_(card)
        return c

    def draw_card(self):
        card = self.deck.get_card()
        return card




    def draw_hand(self):
        h = Hand()
        h.cards = [self.deck.get_card(), self.deck.get_card()]
        h.usableAce = self.usable_ace(h)
        return h


    def usable_ace(self, hand):  # checks hands for a usable ace? **
        h_values = hand.getValues()
        for c in hand.cards:
            if c.name == "Ace":
                if sum(h_values) + 10 <= 21:
                    hand.usableAce = True
                    return True


        hand.usableAce = False
        return False

    def sum_hand(self, hand):  # Return current hand total **
        if self.usable_ace(hand):
            return sum(hand.getValues()) + 10
        return sum(hand.getValues())


    def is_bust(self, hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21


    def score(self, hand):  # What is the score of this hand (0 if bust) **
        score = 0

        if self.is_bust(hand):
            score = 0
        
        else: 
            score = self.sum_hand(hand)
                
        return score
        


    def is_natural(self, hand):  # Is this hand a natural blackjack?
        return sorted(hand.getValues()) == [1, 10]
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False, deck_num=1):
        self.action_space = spaces.Discrete(3)
        self.side_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed(0)

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        self.deck = Deck()

        self.deck.count = 0

        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
            
        if action == 1:  # hit: add a card to players hand and return
            self.playerHand.cards.append(self.draw_card())
            self.sum_hand(self.playerHand)
            if self.is_bust(self.playerHand):
                done = True
                reward = -1
            else:
                done = False
                reward = 0

        elif action == 0:  # stick: play out the dealers hand, and score
            done = True

            while sum(self.dealerHand.getValues()) < 17:
                self.dealerHand.cards.append(self.draw_card())
            reward = self.cmp(self.score(self.playerHand), self.score(self.dealerHand))
        
            if self.is_natural(self.playerHand) and reward == 1:
                if self.is_natural(self.dealerHand):
                    # Push - original bet returned
                    reward = 0
                # natural blackjack pays out 3/2
                reward = 1.5

        elif action == 2: # double: double bet and dealt a final card

            self.playerHand.cards.append(self.draw_card()) 
            self.sum_hand(self.playerHand)


            while sum(self.dealerHand.getValues()) < 17:
                self.dealerHand.cards.append(self.draw_card())

            if self.is_bust(self.playerHand):
                reward = -2
            else:
                reward = self.cmp(self.score(self.playerHand), self.score(self.dealerHand))
                reward *= 2
            done = True

        elif action == "3": # split: split hand
            done = False
            # check for pair before calling
            self.player.append([self.player[0][1]])
            self.player.remove(self.player[0][1])

        return self._get_obs(), reward, done, 0

    def _get_obs(self):
        a = (self.sum_hand(self.playerHand), self.dealerHand.getValues()[0], self.usable_ace(self.playerHand))
        return a

    def get_hands(self):
        return self.playerHand, self.dealerHand

    def get_hand_cards(self):
        return self.playerHand.cards + [self.dealerHand.cards[0]]
        
    def reset(self):
        if (self.deck.last_hand or self.deck.deck_cards.count == 0):
            self.deck.gen_shoe()

        self.dealerHand = self.draw_hand()
        self.usable_ace(self.dealerHand)

        self.playerHand = self.draw_hand()
        self.usable_ace(self.playerHand)
        psum = self.sum_hand(self.playerHand)
        
            
        return self._get_obs()

    def reset_(self):
        self.deck.deck_cards + self.playerHand.cards
        self.deck.deck_cards + self.dealerHand.cards
        self.deck.shuffle_deck()
        self.playerHand = self.draw_hand()
        self.dealerHand = self.draw_hand()

        return self._get_obs()

    def deck_by_count(self, count):
        self.deck.gen_shoe()
        while True:
            c1_deck, c2_deck = self.deck.count_suits
            c1_deck //= self.deck.num_decks
            c2_deck //= self.deck.num_decks
            if tuple([c1_deck, c2_deck]) == count and self.deck.count < (52 * self.deck.num_decks // 2):
                return
            else:
                c1, c2 = zip(count)
                #c1_deck, c2_deck = zip(self.deck.count_suits)
                # check clubs & heart count and adjust
                c1 = c1[0]
                c2 = c2[0]
                if c1 > c1_deck:
                    temp = [card for card in self.deck.deck_cards if card.suit == self.deck.suits[0]]
                    if len(temp) == 0:
                        self.deck.gen_shoe()
                        continue
                    self.draw_card_(np.random.choice(temp))

                if c1 < c1_deck:
                    temp = [card for card in self.deck.deck_cards if card.suit == self.deck.suits[1]]
                    if len(temp) == 0:
                        self.deck.gen_shoe()
                        continue
                    self.draw_card_(np.random.choice(temp))

                if c2 > c2_deck:
                    temp = [card for card in self.deck.deck_cards if card.suit == self.deck.suits[2]]
                    if len(temp) == 0:
                        self.deck.gen_shoe()
                        continue
                    self.draw_card_(np.random.choice(temp))

                if c2 < c2_deck:
                    temp = [card for card in self.deck.deck_cards if card.suit == self.deck.suits[3]]
                    if len(temp) == 0:
                        self.deck.gen_shoe()
                        continue
                    self.draw_card_(np.random.choice(temp))

                