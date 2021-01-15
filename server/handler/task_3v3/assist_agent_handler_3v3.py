# coding:utf-8
import numpy as np
from handler.funcs import *
from handler.agent_handler import AgentHandler

reward_assist = {'lost': -1, 'stealed': -1, 'time_up': -1}


class AssistAgentHandler3v3(AgentHandler):
    def calc_middle_reward(self, now_state):
        r = 0
        # if now_state['game_state']['game_state'] == 'BallClear':
        #     if get_me(now_state)['three_point_area']:
        #         r += 0.1
        return r

    def calc_final_reward(self, ending_object, now_state):
        if not self.pre_state:
            return 0
        game_over_str = ending_object["result"]
        me = get_me(now_state)
        if game_over_str in reward_assist:
            return reward_assist[game_over_str]
        elif game_over_str in ['two', 'three']:
            reward = get_shoot_expect_score(me)
            return reward
        elif game_over_str == "get_ball":
            last_ball_owner = self.pre_state["ball"]["owner_index"]
            expect_last_owner = get_shoot_expect_score(self.pre_state["teams"][now_state["my_team"]][last_ball_owner])
            expect_me = get_shoot_expect_score(me)
            reward = expect_me - expect_last_owner
            me_last = get_me(self.pre_state)
            return reward*2 if me_last['give_me_the_ball'] else 0
        elif game_over_str == "offence_screen":
            return 2 if ending_object["falldown"] else 0
        else:
            raise Exception("unknown game over str %s" % game_over_str)

    def feature_extract(self, now):
        me = get_me(now)
        nearest_opponent = get_nearest_opponent(now)
        ball_owner = get_ball_owner(now)
        opponents = get_opponent(now)
        teammates = get_teammate(now)

        rank = get_player_rank(now)
        result_list = [0] * 3
        result_list[rank] = 1

        for people in [[me], opponents, teammates]:
            if people is None:
                continue
            for member in people:
                result_list.append(member["position"]["x"] / 7.5)
                result_list.append(member["position"]["z"] / 13.78)
                result_list.append(member['basket_distance'] / 12)
                result_list.append(math.sin(math.radians(member["facing"])))
                result_list.append(math.cos(math.radians(member["facing"])))

        result_list.append(float(ball_owner["cannot_dribble"]))
        result_list.append(float(ball_owner["shoot_rate"]) / 100)
        result_list.append(float(me["shoot_rate"]) / 100)
        result_list.append(float(me["three_point_area"]))
        result_list.append(now["game_state"]["attack_remain_time"] / 20)
        result_list.append(distance_of_vector3(me["position"], nearest_opponent["position"]) / 18)
        result_list.append(now["ball"]["position"]["x"] / 7.5)
        result_list.append(now["ball"]["position"]["y"] / 3.8)
        result_list.append(now["ball"]["position"]["z"] / 13.78)
        result_list.extend(get_game_state_one_hot(now["game_state"]))

        return result_list

    def get_available_actions_share(self, now, scene, ending_object):
        """
        For the full game settings.
        """
        state = now['state'] if not ending_object else ending_object['state']
        assert state == 'assist'
        available_actions = np.concatenate((np.ones(11), np.zeros(24)))
        return available_actions
