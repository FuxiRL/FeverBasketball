# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler import AgentHandler

reward_attack = {'two': 2, 'three': 3, 'passed': 0, 'two_blocked': -1, 'three_blocked': -1, 'lost': -1,
                 'pass_lost': -1, 'stealed': -1, 'pass_stealed': -1, 'time_up': -5}


class AttackAgentHandler3v3(AgentHandler):
    def calc_middle_reward(self, now_state):
        r = 0
        # time_delta = get_time_delta(self.pre_state, now_state)
        # r -= time_delta / 10
        # if get_me(now_state)['cannot_dribble']:
        #     r -= 0.1
        return r

    def calc_final_reward(self, ending_object, now_state):
        if not self.pre_state:
            return 0
        game_over_str = ending_object["result"]
        if game_over_str in reward_attack:
            reward = reward_attack[game_over_str]
            if game_over_str in ['two', 'three']:
                left_time = self.pre_state["game_state"]["attack_remain_time"] if self.pre_state else 0
                reward = reward * (ending_object["shoot_percent"] / 100) * (1 - ending_object["block_percent"] / 100.0)
                if reward > 0.5:
                    reward = 2 * reward + left_time/20
                else:
                    reward = - 1
            elif game_over_str == "passed":
                owner_index = now_state["ball"]["owner_index"]
                if self.pre_state and owner_index >= 0:
                    expect_owner = get_shoot_expect_score(self.pre_state["teams"][now_state["my_team"]][owner_index])
                    expect_me = get_shoot_expect_score(get_me(self.pre_state))
                    reward = expect_owner - expect_me
                    if reward > 0 and self.pre_state["teams"][now_state["my_team"]][owner_index]["give_me_the_ball"]:
                        reward *= 2
                left_time = now_state["game_state"]["attack_remain_time"]
                if left_time <= 5:
                    reward -= 5
            return reward
        else:
            print("unknown game over str %s" % game_over_str)
            return 0

    def feature_extract(self, now):
        me = get_me(now)
        nearest_opponent = get_nearest_opponent(now)

        opponents = get_opponent(now)
        teammates = get_teammate(now)

        rank = get_player_rank(now)
        result_list = [0] * 3
        result_list[rank] = 1

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
        result_list.append(float(me["shoot_rate"]) / 100)
        result_list.append(float(me["three_point_area"]))
        result_list.append(now["game_state"]["attack_remain_time"] / 20)
        result_list.append(distance_of_vector3(me["position"], nearest_opponent["position"]) / 18)
        result_list.extend(get_game_state_one_hot(now["game_state"]))

        return result_list

    def get_available_actions_share(self, now, scene, ending_object):
        """
        For the full game settings.
        """
        state = now['state'] if not ending_object else ending_object['state']
        assert state == 'attack'
        available_actions = np.ones(35)
        return available_actions