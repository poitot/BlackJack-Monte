import numpy as np
import random

class Card:
    name = ""
    suit = ""
    value = 0
    cut = False

    def __init__(self, name, suit, value, cut):
        self.name = name
        self.suit = suit
        self.value = value
        self.cut = cut

class Hand:
    def __init__(self):
        self.cards = []
        self.splitable = False
        self.usableAce = False
        self.done = False

    def getValues(self):
        values = []
        for card in self.cards:
            values.append(card.value)

        return values

    def checkSplit(self):
        if (self.cards[0].value == self.cards[1].value):
            self.splitable = True

    def split(self):
        if (self.splitable):
            hand1 = [cards[0]]
            hand2 = [cards[1]]
            return hand1, hand2


class Deck:

    num_decks = 4

    def __init__(self):
        self.cards = [["Ace", 1], ["Two", 2], ["Three", 3], ["Four", 4], ["Five", 5], ["Six", 6], ["Seven", 7], ["Eight", 8], ["Nine", 9], ["Ten", 10], ["Jack", 10], ["Queen", 10], ["King", 10]]
        self.suits = {"Clubs", "Diamonds", "Hearts", "Spades"}
        self.deck_cards = []
        self.last_hand = False
        self.count = 0

    def gen_deck(self):

        for card in self.cards:
            for suit in self.suits:
                card_temp = Card(card[0], suit, card[1], False)
                self.deck_cards.append(card_temp)

    def add_cut(self):
        cut_card = Card(0, "Cut", 0, True)
        if (cut_card not in self.deck):
            self.deck.append(cut_card)
        

    def gen_shoe(self):
        self.deck_cards = []
        self.count = 0
        shoe = Deck()
        for x in range(self.num_decks):
            self.gen_deck()
            for c in self.deck_cards:
                shoe.deck_cards.append(c)

        # add cut card to shoe before returning it
        # cut card is placed randomly in the shoe and when drawn signals that the shoe will be re-made for the next hand
        cut_card = Card("cut", "none", 0, True)
        shoe.deck_cards.append(cut_card)
        shoe.shuffle_deck()
        return shoe

    def shuffle_deck(self):
        random.shuffle(self.deck_cards)
        

    def get_card(self):
        if len(self.deck_cards) > 0:
            temp_card = self.deck_cards.pop(0)

        else:
            self.gen_shoe()
            temp_card = self.deck_cards.pop(0)

        # if cut card is drawn give a new card to the player & set lastHand flag
        if temp_card.cut:
            temp_card = self.deck_cards.pop(0)
            self.last_hand

        """ updates deck count as cards are dealt
            cards 2-6  : +1
            cards 7-9  : +0
            cards 10-A : -1

            will be used with number of decks in shoe to determine a true count of the deck
            to be added to observations
        """
        
        if 1 < temp_card.value < 7:
            self.count += 1
        elif 9 < temp_card.value or temp_card.name == "Ace":
            self.count -= 1

        return temp_card

    # returns the true count of a shoe
    def get_true_count(self):
        return self.count // self.num_decks


    def deal_hand(self):
        if self.last_hand:
            self.gen_shoe()
            self.shuffle_deck()

        card_1 = self.get_card()
        card_2 = self.get_card()

        hand = [card_1, card_2]
        return hand