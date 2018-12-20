from game.liarsdice import *
from agent.smart_ai import *
from game.interaction import *
from game.liarsdice import *
from util import *

def get_target_policies(prev_bids, players_dice, agent, policy_prob):
    next_state_values = agent.get_next_state_values(prev_bids, players_dice)
    probs = np.asarray(policy_prob)
    target_policies = np.zeros(probs.shape)
    for i in range(0, len(policy_prob)):
        start_ind = np.where(policy_prob[i] != 0)[0][0]
        next_values_array = np.asarray(next_state_values[i])
        e = np.dot(policy_prob[i][start_ind:], next_values_array)
        E = next_values_array - e
        target_policies[i][start_ind:] = policy_prob[i][start_ind:] + E
    return np.asarray(target_policies)

def get_target_values(payoff, history):
    cur_player = 1-(len(history) % 2)
    gamma = 0.9
    # if len(history) == 2:
    #     return [payoff[0],payoff[0]]
    history = history[::-1]

    target_values = []
    target_values.append(payoff[cur_player])
    cur_player = 1-cur_player
    target_values.append(payoff[cur_player])

    for i in range(2, len(history)):
        target_values.append(gamma*target_values[-2] + 0)

    target_values = target_values[::-1]
    return np.asarray(target_values)

if __name__ == '__main__':
    save_loss_each = 100
    save_model_each = 1000
    max_iter = 100000000
    player = smart_agent('Jared')
    game = liars_dice_game()
    game.start_game()
    policy_losses = []
    value_losses = []
    policy_loss_sum = 0
    value_loss_sum = 0
    for i in range(0, max_iter):
        pay_off, traj, o_player1, o_player2, policy_probs, masks = run_game(player, game)
        prev_bids = get_prev_bids(traj)
        prev_bids.append('checked')
        prev_bids.append(traj[-1])
        policy_targets = get_target_policies(prev_bids[:-2], [o_player1, o_player2], player, policy_probs)
        states = get_states(prev_bids[:-2], [o_player1, o_player2])
        policy_loss = player.train_policy_net(states, policy_targets, masks)
        Value_targets = get_target_values(pay_off, prev_bids)
        value_states = get_value_states(prev_bids, [o_player1, o_player2])
        value_loss, values, entry_diffs = player.train_Value_net(value_states, Value_targets)
        print('prev_bids:{}\nentry_diff:{}\n Value_targets:{}\n'.format(prev_bids, entry_diffs, Value_targets))

        policy_loss_sum += policy_loss
        value_loss_sum += value_loss
        # if not (i+1)%save_loss_each:
        #     policy_losses.append(policy_loss/save_loss_each)
        #     value_losses.append(value_loss/save_loss_each)
        print('policy_loss is: {}, value_loss is {}'.format(policy_loss, value_loss))
        if not (i+1)% save_model_each:
            player.saver.save(player.sess, 'model_save/model.ckpt', global_step=i)
