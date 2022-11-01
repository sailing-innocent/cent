import collections
from random import choice

Card = collections.namedtuple('Card', ['rank', 'suit'])

# namedtuple was introduced into python since v2.6 for creating some classes with a few attributes but no methods
# like Database Index

class FrenchDeck:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]


# beer_card = Card('7', 'diamonds')
# print(beer_card)

deck = FrenchDeck()

# Then you can use it as a common object
# print(len(deck))
# print(choice(deck))
# for card in deck:
# for card in reversed(deck)
# if a set haven't realize its own __contains__ method, then it will iterate once to search for it 
print(Card('Q','hearts') in deck)
    
# for in in x: x.__iter__() method
