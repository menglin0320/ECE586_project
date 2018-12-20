from util import *

class dumb_ai_for_test:
    def __init__(self, name):
        self.name = name
        self.bid_seen = (0, 0)
        self.all_actions = get_all_actions()

    def see_dice(self, dices):
        self.dices = dices

    def see_bid(self, others_bid):
        self.bid_seen = others_bid

    def get_valid_actions(self):
        num, face = self.bid_seen
        if num == 0 and face == 0:
            return self.all_actions
        else:
            return self.all_actions[((num - 1) * 6 + face - 1) + 1:]

    def bid(self):
        all_valid_actions = self.get_valid_actions()
        ind = random.randint(0, len(all_valid_actions) - 1)
        return all_valid_actions[ind]