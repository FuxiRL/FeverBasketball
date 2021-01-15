# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler_ma import AgentHandler

reward_ballclear = {'success': 1, 'passed': 1, 'get_ball': 1, 'lost': -1, 'pass_lost': -1, 'stealed': -1,
                    'pass_stealed': -1, 'time_up': -5}


class BallclearAgentHandler3v3_ma(AgentHandler):
    def calc_middle_reward(self, now_state):
        reward = 0
        # if not self.pre_state:
        #     return 0
        # me_pre = get_me(self.pre_state)
        # me_now = get_me(now_state)
        # # # last_action = now_state["last_action"]
        # time_delta = get_time_delta(self.pre_state, now_state)
        # reward = (me_now["basket_distance"] - me_pre["basket_distance"]) / 10
        # # if reward == 0:
        # #     reward -= 0.1
        # reward_time = time_delta / 10
        # reward -= reward_time
        return reward

    def calc_final_reward(self, ending_object, now_state):
        if not self.pre_state:
            return 0

        game_over_str = ending_object["result"]
        if game_over_str in reward_ballclear:
            reward = reward_ballclear[game_over_str]
            if ending_object["result"] == "success":
                left_time = now_state["game_state"]["attack_remain_time"]
                reward += left_time / 20
            return reward
        else:
            print("unknown game over str %s" % game_over_str)
            return 0

    def feature_extract_local(self, now, scene, index):
        assert scene == 'ballclear'
        state = now['state']
        me = get_me(now)
        nearest_opponent = get_nearest_opponent(now)

        opponents = get_opponent(now)
        teammates = get_teammate(now)

        rank = get_player_rank(now)
        result_list = [0] * 3
        result_list[rank] = 1

        # add scene onehot [ballclear, assist]
        if state == 'ballclear':
            result_list.extend([1, 0])  # ballclear scene includes both 'ballclear' state and 'assist' within the team
        elif state == 'assist':
            result_list.extend([0, 1])

        # add index onehot
        index_onehot = [0] * 3
        index_onehot[index] = 1
        result_list.extend(index_onehot)

        for people in [[me], opponents, teammates]:
            for member in people:
                result_list.append(member["position"]["x"] / 7.5)
                result_list.append(member["position"]["z"] / 13.78)
                result_list.append(member['basket_distance'] / 12)
                result_list.append(math.sin(math.radians(member["facing"])))
                result_list.append(math.cos(math.radians(member["facing"])))
        for member in teammates:
            result_list.append(member["give_me_the_ball"])
            result_list.append(member["shoot_rate"] / 100)

        result_list.append(float(me["cannot_dribble"]))
        result_list.append(float(me["basket_distance"] / 18))
        result_list.append(float(me["shoot_rate"]) / 100)
        result_list.append(float(me["three_point_area"]))
        result_list.append(now["game_state"]["attack_remain_time"] / 20)
        result_list.append(distance_of_vector3(me["position"], nearest_opponent["position"]) / 18)
        result_list.append(now["ball"]["position"]["x"] / 7.5)
        result_list.append(now["ball"]["position"]["y"] / 3.8)
        result_list.append(now["ball"]["position"]["z"] / 13.78)
        result_list.extend(get_game_state_one_hot(now["game_state"]))

        return result_list

    def get_available_actions(self, now, scene, ending_object):
        state = now['state'] if not ending_object else ending_object['state']
        assert scene == 'ballclear'
        if state == 'ballclear':
            available_actions = np.ones(25)
        else:
            assert state == 'assist'
            available_actions = np.concatenate((np.ones(11), np.zeros(14)))
        return available_actions

    def get_available_actions_share(self, now, scene, ending_object):
        state = now['state'] if not ending_object else ending_object['state']
        assert scene == 'ballclear'
        if state == 'ballclear':
            available_actions = np.concatenate((np.ones(25), np.zeros(10)))
        else:
            assert state == 'assist'
            available_actions = np.concatenate((np.ones(11), np.zeros(24)))
        return available_actions






