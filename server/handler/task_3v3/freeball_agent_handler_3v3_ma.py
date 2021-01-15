# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler_ma import AgentHandler

reward_freeball = {'me': 1, 'opponent': -1, 'goal_in': 0, 'time_up_me': -1, 'time_up_opponent': -1}


class FreeballAgentHandler3v3_ma(AgentHandler):
    def calc_middle_reward(self, now_state):
        r = 0
        # data_me_now = get_me(now_state)
        # data_me_pre = get_me(self.pre_state)
        # dis_ball_me_pre = distance_of_vector2(self.pre_state["ball"]["position"], data_me_pre["position"])
        # dis_ball_me_now = distance_of_vector2(now_state["ball"]["position"], data_me_now["position"])
        # r = (dis_ball_me_pre - dis_ball_me_now) / 10
        # print('freeball middle r', r)
        return r

    def calc_final_reward(self, ending_object, now_state):
        game_over_str = ending_object["result"]
        if game_over_str in reward_freeball:
            reward = reward_freeball[game_over_str]
            return reward
        else:
            print("unknown game over str %s" % game_over_str)
            return 0

    def feature_extract_local(self, now, scene, index):
        me = get_me(now)
        nearest_opponent = get_nearest_opponent(now)
        nearest_teammate = get_nearest_teammate(now)
        opponents = get_opponent(now)
        teammates = get_teammate(now)
        rank = get_player_rank(now)
        result_list = [0] * 3
        result_list[rank] = 1

        # add index onehot
        index_onehot = [0] * 3
        index_onehot[index] = 1
        result_list.extend(index_onehot)

        for people in [[me], opponents, teammates]:
            if people is None:
                continue
            for member in people:
                result_list.append(member["position"]["x"] / 7.5)
                result_list.append(member["position"]["z"] / 13.78)
                result_list.append(member['basket_distance'] / 12)
                result_list.append(math.sin(math.radians(member["facing"])))
                result_list.append(math.cos(math.radians(member["facing"])))

        result_list.append(now["game_state"]["attack_remain_time"] / 20)
        result_list.append(distance_of_vector3(me["position"], nearest_opponent["position"]) / 18)
        result_list.append(distance_of_vector3(now["ball"]["position"], me["position"]) / 18)
        result_list.append(float(me["is_attacking_team"]))
        result_list.append(now["ball"]["position"]["x"] / 7.5)
        result_list.append(now["ball"]["position"]["y"] / 3.8)
        result_list.append(now["ball"]["position"]["z"] / 13.78)

        return result_list

    def get_available_actions(self, now, scene, ending_object):
        assert scene == 'freeball'
        available_actions = np.ones(9)
        return available_actions

    def get_available_actions_share(self, now, scene, ending_object):
        assert scene == 'freeball'
        available_actions = np.concatenate((np.ones(9), np.zeros(26)))
        return available_actions

