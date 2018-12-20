import numpy as np
import random

def get_all_actions():
    ret_list = []
    for i in range(0, 10):
        for j in range(0, 6):
            ret_list.append([i+1, j+1])
    ret_list.append('liar')
    return ret_list

def epsilan_greedy(valid_actions, best_action, k, epsilon):
    epsilon = max(0, epsilon - k/1000000)
    do_rand_walk = np.random.choice([False,True],1, p = [1-epsilon, epsilon])
    if do_rand_walk:
        action = valid_actions[random.randint(0, len(valid_actions)-1)]
        return action
    else:
        if best_action == 60:
            return 'liar'
        return [best_action// 6 + 1, best_action % 6 + 1]

def action2onehot(action):
    max_dices = 10
    max_faces = 6
    action_size = max_dices * max_faces + 1
    one_hot = np.zeros(action_size)

    if action[0] == 0:
        one_hot[- 1] = 1
        return one_hot
    ind = (action[0]-1) * max_faces + (action[1]-1)
    one_hot[ind-1] = 1
    return one_hot

def value_action2onehot(action):
    max_dices = 10
    max_faces = 6
    action_size = max_dices * max_faces + 3
    one_hot = np.zeros(action_size)

    if action == 'liar':
        one_hot[-3] = 1
        return one_hot

    if action == 'checked':
        one_hot[-2] = 1
        return one_hot

    if action[0] == 0:
        one_hot[- 1] = 1
        return one_hot

    ind = (action[0]-1) * max_faces + (action[1]-1)
    one_hot[ind-1] = 1
    return one_hot

def dice2onehot(dice):
    max_dices = 5
    max_faces = 6
    state_size = max_dices * max_faces
    dice_array = np.array(dice)
    ind =  np.sum(dice_array) - 1
    one_hot = np.zeros(state_size)
    one_hot[ind] = 1
    return one_hot

def to_histogram(game_dices):
    hist = np.zeros((11))
    for i in range(0, len(game_dices)):
        for dice in range(0, len(game_dices)):
            hist[game_dices[i][dice]] += 1
    return hist

def get_valid_actions(all_actions, current_bid):
    num, face = current_bid
    if num == 0 and face == 0:
        return all_actions
    else:
        return all_actions[((num - 1) * 6 + face - 1) + 1:]

def get_prev_bids(history):
    prev_bids = [(0, 0)]
    prev_bids.extend(history[:-1])
    return prev_bids

def get_state(bid, dice):
    one_hot_bid = action2onehot(bid)
    one_hot_seen = dice2onehot(dice)
    whole_state_in = np.expand_dims(np.concatenate((one_hot_bid, one_hot_seen), axis=0), 0)
    return whole_state_in

def get_value_state(bid, dice):
    value_hot_bid = value_action2onehot(bid)
    one_hot_seen = dice2onehot(dice)
    whole_state_in = np.expand_dims(np.concatenate((value_hot_bid, one_hot_seen), axis=0), 0)
    return whole_state_in

def get_states(previous_bids, players_dice):
    game_batch = []
    cur_player = 1
    for i in range(len(previous_bids)):
        cur_player = 1-cur_player
        game_batch.append(get_state(previous_bids[i],players_dice[cur_player])[0])
    return np.asarray(game_batch)

def get_value_states(previous_bids, players_dice):
    game_batch = []
    cur_player = 1
    for i in range(len(previous_bids)):
        cur_player = 1 - cur_player
        game_batch.append(get_value_state(previous_bids[i], players_dice[cur_player])[0])
    return np.asarray(game_batch)

def int2action(probs):
    max_ind = np.argmax(probs)
    if max_ind == 60:
        return 'liar'
    else:
        return [max_ind//6 + 1, max_ind%6 + 1]

def before_liar2int(before_liar):
    if before_liar[0] == 0 and before_liar[1] == 0:
        return 62
    else:
        return 6*(before_liar[0] - 1) + before_liar[1] - 1
