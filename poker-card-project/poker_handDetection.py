import ultralytics
import numpy as np


def findPokerHand(hand):
    ranks = []
    suits = []
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        ranks.append(rank)
        suits.append(suit)

    # checking for flush or 5 element of the suits case are the same
    if suits.count(suits[0]) == 5:





    return 0

if __name__ == "__main__":
    findPokerHand(['AH', 'KH', 'QH', 'JH', '10H']) #ROYAL FLUSH
    findPokerHand(['QC', 'JC', '10C', '9C', '8C']) #STRAIGHT FLUSH
    findPokerHand(['5C', '5H', '5S', '10D', '7S']) #THREE OF A KIND
    findPokerHand(['2H', '5H', 'QH', 'JH', '8H']) #FLUSH

