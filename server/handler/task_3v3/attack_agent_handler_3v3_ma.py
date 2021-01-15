# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler_ma import AgentHandler

reward_attack = {'two': 2, 'three': 3, 'passed': 0, 'two_blocked': -1, 'three_blocked': -1, 'lost': -1,
                 'pass_lost': -1, 'stealed': -1, 'pass_stealed': -1, 'time_up': -5}


class AttackAgentHandler3v3_ma(AgentHandler):
    def calc_middle_reward(self, now_state):
        r = 0
        # time_delta = get_time_delta(self.pre_state, now_state)
        # r -= time_delta / 10
        # if get_me(now_state)['cannot_dribble'] and now_state['state'] == 'attack':
        #     r -= 0.1
        return r

    def calc_final_reward(self, ending_object, now_state):
        if not self.pre_state:
            return 0
        game_over_str = ending_object["result"]
        if game_over_str in reward_attack:
            reward = reward_attack[game_over_str]
            if game_over_str in ['two', 'three']:
                if ending_object['state'] == 'attack':
                    left_time = self.pre_state["game_state"]["attack_remain_time"] if self.pre_state else 0
                    reward = reward * (ending_object["shoot_percent"] / 100) * (1 - ending_object["block_percent"] / 100.0)
                    if reward > 0.5:
                        reward = 2 * reward + left_time/20
                    else:
                        reward = - 1
                else:
                    reward = 0  # we don't directly reward assist, need to learn according to team reward.
            return reward
        else:
            print("unknown game over str %s" % game_over_str)
            return 0

    def feature_extract_local(self, now, scene, index):
        assert scene == 'attack'
        state = now['state']
        me = get_me(now)
        nearest_opponent = get_nearest_opponent(now)

        opponents = get_opponent(now)
        teammates = get_teammate(now)

        rank = get_player_rank(now)
        result_list = [0] * 3
        result_list[rank] = 1

        # add state onehot [attack, assist]
        if state == 'attack':
            result_list.extend([1, 0])  # attack scene includes both 'attack' state and 'assist' within the team
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
        """
        For the divide_and_conquer settings.
        """
        state = now['state'] if not ending_object else ending_object['state']
        assert scene == 'attack'
        if state == 'attack':
            available_actions = np.ones(35)
        else:
            assert state == 'assist'
            available_actions = np.concatenate((np.ones(11), np.zeros(24)))
        return available_actions

    def get_available_actions_share(self, now, scene, ending_object):
        """
        For the full game settings.
        """
        state = now['state'] if not ending_object else ending_object['state']
        assert scene == 'attack'
        if state == 'attack':
            available_actions = np.ones(35)
        else:
            assert state == 'assist'
            available_actions = np.concatenate((np.ones(11), np.zeros(24)))
        return available_actions

