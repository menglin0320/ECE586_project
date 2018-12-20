
from util import *

class liars_dice_game():
    def __init__(self, n_player=2, n_dices=5, n_faces=6):
        self.n_player = n_player
        self.n_dices = n_dices
        self.n_faces = n_faces
        self.dices = [[0] * 5]*2
        self.history = [[],[]]

    def roll_dices(self):
        self.dices = []
        for i in range(self.n_player):
            self.dices.append([])
            for j in range(self.n_dices):
                self.dices[i].append(random.randint(1, self.n_faces))

    def judge(self, last_player_ind, history):
        dice_hist = to_histogram(self.dices)
        if not history[1]:
            return (-1,1)
        last_bid = history[1-last_player_ind][-1]
        if dice_hist[last_bid[1]] >= last_bid[0] and last_player_ind == 0 or\
            dice_hist[last_bid[1]] < last_bid[0] and last_player_ind == 1:
            return (-1,1)
        else:
            return (1,-1)

    def start_game(self):
        self.roll_dices()
        self.history = [[],[]]

    def peek(self, player_id):
        return self.dices[player_id]

    def get_last_bid(self, player_id):
        if not self.history[0]:
            return (0,0)
        else:
            return self.history[1-player_id][-1]

    def take_action(self, player_id, chosen_action):
        if(self.history[-1] == 'liar'):
            print('game already end, please restart the game if you want to play a new turn')
            return (0,0)
        self.history[player_id].append(chosen_action)
        payoff = [0,0]
        if self.history[player_id][-1] == 'liar':
            payoff = self.judge(player_id, self.history)
        return payoff

