from agent.dumb_ai import *
from game.liarsdice import *

if __name__ == '__main__':
    player_0 = dumb_ai_for_test(0)
    player_1 = dumb_ai_for_test(1)
    players = [player_0, player_1]
    game = liars_dice_game()
    for j in range(0, 10):
        pay_off = [0, 0]
        game.start_game()
        player_0.see_dice(game.peek(player_0.name))
        player_1.see_dice(game.peek(player_1.name))
        i = 1
        while (pay_off == [0, 0]):
            i = 1 - i
            state = game.get_last_bid(i)
            players[i].see_bid(state)
            pay_off = game.take_action(i, players[i].bid())
        print('payoff_matrix is: {}\n player1 bids: {}\n, player2 bids: {},\n player1 dices {}\n player2 dices {}\n\n'
              .format(pay_off, game.history[0], game.history[1], game.dices[0], game.dices[1]))