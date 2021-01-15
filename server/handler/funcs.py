# coding: utf-8
"""
Useful functions for state and reward generation.
"""
import math

RIM_POINT = {"x": 0.0, "y": 3.02, "z": 12.08}

# state space from multi-agent perspective
state_dim_global = 58
# for full game mode
state_space_full = 66
# for divide_and_conquer mode
state_space_map = {'attack': 56,     # attack, assist, assist
                   'defense': 27,    # defense, defense, defense
                   'freeball': 43,   # freeball, freeball, freeball
                   'ballclear': 56}  # ballclear, assist, assist

# action space for SG player
# for full game mode
action_space_full = 35
# for divide_and_conquer mode
action_space_map = {'attack': 35,
                    'defense': 27,
                    'freeball': 9,
                    'ballclear': 25,}

def get_scene(state, game_state):
    """
    Can be used to get the scene name if the divided_and_conquer mode is considered under multi-agent perspective.
    :param state:
    :param game_state:
    :return:
    """
    scene = None
    if state == 'attack':
        scene = 'attack'
    elif state == 'defense':
        scene = 'defense'
    elif state == 'freeball':
        scene = 'freeball'
    elif state == 'ballclear':
        scene = 'ballclear'
    elif state == 'assist':
        scene = 'ballclear' if game_state == 'BallClear' else 'attack'
    else:
        print('Scene:{} game_state:{} is not considered!!!'.format(state, game_state))
    return scene


def get_me(now, index=None):
    my_team = now["my_team"]
    my_index = now["my_index"] if index is None else index
    return now["teams"][my_team][my_index]


def get_opponent(now, index=None):
    rel = []
    my_team = now["my_team"]
    my_index = now["my_index"] if index is None else index
    for i in range(len(now["teams"])):
        for j in range(len(now["teams"][i])):
            if i != my_team: rel.append(now["teams"][i][j])
    return rel


def get_teammate(now, index=None):
    rel = []
    my_team = now["my_team"]
    my_index = now["my_index"] if index is None else index
    for i in range(len(now["teams"][my_team])):
        if i != my_index: rel.append(now["teams"][my_team][i])

    return rel


def get_ball_owner(now, index=None):
    if now["ball"]["owner_index"] < 0 or now["ball"]["team"] < 0:
        return get_nearest_opponent(now, index)
    else:
        team_index = now["ball"]["team"]
        owner_index = now["ball"]["owner_index"]
        return now["teams"][team_index][owner_index]


def get_state_3v3defend(now_state, index=None):
    result = {
        "me": get_me(now_state, index),
        "opponent": get_match_opponent(now_state, index),
        "teammate": get_nearest_teammate(now_state, index),
        "owner": get_ball_owner(now_state, index),
    }
    for k, v in now_state.items():
        result[k] = v
    return result


def get_nearest_opponent(now, index=None):
    me = get_me(now, index)
    opponents = get_opponent(now, index)
    opponents = [opponents] if not isinstance(opponents, list) else opponents

    _index = None
    min_dist = 1000000.0
    for i in range(len(opponents)):
        d = distance_of_vector3(me["position"], opponents[i]["position"])
        if d < min_dist:
            _index = i
            min_dist = d

    return opponents[_index]


def get_nearest_teammate(now, index=None):
    me = get_me(now, index)
    teammates = get_teammate(now, index)
    # teammates = [teammates] if not isinstance(teammates, list) else teammates

    _index = None
    min_dist = 1000000.0
    for i in range(len(teammates)):
        d = distance_of_vector3(me["position"], teammates[i]["position"])
        if d < min_dist:
            _index = i
            min_dist = d

    if _index is None:
        return None
    return teammates[_index]


def get_match_opponent(now, index=None, force_same_index=False):
    my_team = now["my_team"]
    my_index = now["my_index"] if index is None else index
    defense_strategy = now["defense_strategy"]
    if now["text"] == "defense_training" or defense_strategy < 0 or force_same_index:
        return now["teams"][1-my_team][my_index]
    else:
        tmp = defense_strategy
        for i in range(my_index):
            tmp //= 3
        return now["teams"][1-my_team][tmp%3]


def get_type_one_hot(person):
    result = [0] * 5
    rank = {
        "PG": 0,
        "SG": 1,
        "SF": 2,
        "PF": 3,
        "C": 4,
    }
    result[rank[person["type"]]] = 1
    return result


REBOUNDER_VALUE = {
    "C": 5,
    "PF": 4,
    "SF": 3,
    "SG": 2,
    "PG": 1,
}
SHOOTER_VALUE = {
    "SG": 5,
    "SF": 4,
    "PG": 3,
    "PF": 2,
    "C": 1,
}


def find_best_player(VALUE_MAP, team):
    best_index = -1
    for i in range(len(team)):
        if best_index == -1 or VALUE_MAP[team[i]["type"]] > VALUE_MAP[team[best_index]["type"]]:
            best_index = i
    return best_index


# return the rank of the player, 0: rebounder, 1: shooter, 2: guard
def get_player_rank(now, index=None):
    team = now["teams"][now["my_team"]][:]
    if len(team) == 1:
        return 2
    me = get_me(now, index)
    best_rebounder_index = find_best_player(REBOUNDER_VALUE, team)
    if team[best_rebounder_index] is me:
        return 0
    team.pop(best_rebounder_index)
    if len(team) == 1:
        return 2
    best_shooter_index = find_best_player(SHOOTER_VALUE, team)
    if team[best_shooter_index] is me:
        return 1
    return 2


def get_shoot_expect_score(person):
    score = 2
    if person["three_point_area"]:
        score = 3
    return person["shoot_rate"] * score / 100.0


def get_game_state_one_hot(game_state):
    result = [0] * 5
    rank = {
        "AttackReady": 0,
        "BallClear": 1,
        "Playing": 2,
        "GoalIn": 3,
        "ShotclockViolation": 4,
    }
    result[rank[game_state["game_state"]]] = 1
    return result


def distance_of_vector3(me, opponent):
    sqr_sum = 0.0
    for attr in ["x", "y", "z"]:
        sqr_sum += (me[attr] - opponent[attr]) ** 2
    return math.sqrt(sqr_sum)


def distance_of_vector2(me, opponent):
    sqr_sum = 0.0
    for attr in ["x", "z"]:
        sqr_sum += (me[attr] - opponent[attr]) ** 2
    return math.sqrt(sqr_sum)


def get_time_delta(pre_state, now_state):
    return now_state["game_state"]["game_local_time"] - pre_state["game_state"]["game_local_time"]







