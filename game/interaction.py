from game.liarsdice import *
from agent.smart_ai import *

def interleave_arrays(a,b):
    interleaved = []
    for i in range(len(a) + len(b)):
        if not i%2:
            interleaved.append(a[i//2])
        else:
            interleaved.append(b[i//2])
    return interleaved


def run_game(players, game):
    pay_off = [0, 0]
    game.start_game()
    i = 1
    mix_policy = []
    masks = []
    while (pay_off == [0, 0]):
        i = 1 - i
        players.see_dice(game.peek(i))
        last_bid = game.get_last_bid(i)
        players.see_bid(last_bid)
        action_probs, action, mask = players.act(players.bid, players.dice)
        pay_off = game.take_action(i, action)
        # print(action_probs)
        # print (pay_off, action)
        mix_policy.append(action_probs)
        masks.append(mask[0])
    print('payoff_matrix is: {}\n player1 bids: {}\n, player2 bids: {},\n player1 dices {}\n player2 dices {}\n\n'
          .format(pay_off, game.history[0], game.history[1], game.dices[0], game.dices[1]))
    seq_history = interleave_arrays(game.history[0], game.history[1])
    return pay_off, seq_history, game.dices[0], game.dices[1], mix_policy, masks

if __name__ == '__main__':
    players = smart_agent('Jared')
    game = liars_dice_game()
    game.start_game()
    run_game(players, game)