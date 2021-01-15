# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler_ma import AgentHandler

reward_defense = {'two_blocked': 2, 'three_blocked': 3, 'lost': 1, 'stealed': 1, 'time_up': 1}


class DefenseAgentHandler3v3_ma(AgentHandler):
    def calc_middle_reward(self, now_state):
        r = 0
        # if self.pre_state:
        #     data_me_pre = get_me(self.pre_state)
        #     data_me_now = get_me(now_state)
        #     ball_owner = get_ball_owner(now_state)
        #     match_oppo_pre_data = get_match_opponent(self.pre_state)
        #     match_oppo_now_data = get_match_opponent(now_state)
        #
        #     # shaping reward for approaching oppo
        #     dis_oppo_me_pre = distance_of_vector2(match_oppo_pre_data["position"], data_me_pre["position"])
        #     dis_oppo_me_now = distance_of_vector2(match_oppo_now_data["position"], data_me_now["position"])
        #     r_dis = dis_oppo_me_pre - dis_oppo_me_now
        #
        #     # shaping reward for standing in right angle
        #     me_dx = data_me_pre["position"]["x"] - 0
        #     me_dz = data_me_pre["position"]["z"] - 12.08
        #     opp_dx = match_oppo_pre_data["position"]["x"] - 0
        #     opp_dz = match_oppo_pre_data["position"]["z"] - 12.08
        #     me_dr = math.sqrt(me_dx ** 2 + me_dz ** 2)
        #     opp_dr = math.sqrt(opp_dx ** 2 + opp_dz ** 2)
        #     cos = (me_dx * opp_dx + me_dz * opp_dz) / (me_dr * opp_dr)
        #     r_cos = 0
        #     if opp_dr >= me_dr:
        #         if cos >= 0.9:
        #             r_cos += 0.1  # 0
        #         elif cos > 0.8:
        #             r_cos += 0.05
        #         else:
        #             r_cos -= 0.1  # 0
        #     else:
        #         r_cos -= 0.2  # 0
        #     #
        #     # # shaping reward for oppo shoot rate decrease
        #     r_shoot_rate = (match_oppo_pre_data["shoot_rate"] - match_oppo_now_data["shoot_rate"]) / 100
        #
        #     # # shaping reward for making oppo can not dribble
        #     r_oppo_dribble_fail = 0
        #     if ball_owner["cannot_dribble"]:
        #         r_oppo_dribble_fail = 0.1
        #
        #     # # pushing for steal improperly
        #     # r_punish = 0
        #     # if 'last_action' in data:
        #     #     last_action = data['last_action']
        #     #     if not data_me_pre['can_steal'] and last_action['name'] == 'Steal':
        #     #         r_punish = - 0.5
        #
        #     r = (r_dis + r_cos) / 2 + (r_shoot_rate + r_oppo_dribble_fail) / 2

        return r

    def calc_final_reward(self, ending_object, now_state):
        if not self.pre_state:
            return 0
        game_over_str = ending_object["result"]
        if game_over_str in reward_defense:
            pre_state = get_state_3v3defend(self.pre_state)
            pre_owner_index = pre_state["ball"]["owner_index"]
            # match player with opponent
            if pre_owner_index >= 0 and pre_state["owner"] is pre_state["opponent"]:
                return reward_defense[game_over_str]
            else:
                return 0

        elif game_over_str in {"two", "three"}:
            score = 2
            if game_over_str.startswith("three"):
                score = 3
            expect = score * (ending_object["shoot_percent"] / 100.0) * (1 - ending_object["block_percent"] / 100.0)
            reward = -expect * 2 if expect > 0.5 else 1
            match_opponent = get_match_opponent(self.pre_state)
            if self.pre_state["teams"][1-self.pre_state["my_team"]][ending_object["shooter_index"]] is match_opponent:
                return reward
            else:
                return 0
        else:
            print("unknown game over str %s" % game_over_str)
            return 0

    def feature_extract_local(self, now, scene, index):
        result_list = []
        me = get_me(now)
        match_opponent = get_match_opponent(now)
        ball_owner = get_ball_owner(now)

        # add index onehot
        index_onehot = [0] * 3
        index_onehot[index] = 1
        result_list.extend(index_onehot)

        for people in [me, match_opponent]:
            result_list.append(people["position"]["x"] / 7.5)
            result_list.append(people["position"]["z"] / 13.78)
            result_list.append(people['basket_distance'] / 12)
            result_list.append(math.sin(math.radians(people["facing"])))
            result_list.append(math.cos(math.radians(people["facing"])))

        result_list.append(float(match_opponent["cannot_dribble"]))
        result_list.append(float(match_opponent["shoot_rate"]) / 100)
        result_list.append(float(ball_owner is match_opponent))

        result_list.append(now["game_state"]["attack_remain_time"] / 20)
        result_list.append(distance_of_vector3(me["position"], match_opponent["position"]) / 18)
        result_list.append(float(me["can_steal"]))
        result_list.append(now["ball"]["position"]["x"] / 7.5)
        result_list.append(now["ball"]["position"]["y"] / 3.8)
        result_list.append(now["ball"]["position"]["z"] / 13.78)
        result_list.extend(get_game_state_one_hot(now["game_state"]))
        return result_list

    def get_available_actions(self, now, scene, ending_object):
        assert scene == 'defense'
        available_actions = np.ones(27)
        return available_actions

    def get_available_actions_share(self, now, scene, ending_object):
        assert scene == 'defense'
        available_actions = np.concatenate((np.ones(27), np.zeros(8)))
        return available_actions

