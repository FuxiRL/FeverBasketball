#coding: utf-8
"""
The base class of AgentHandler, which is used to generate experience for multi-agent perspective.
"""
from abc import abstractmethod, ABCMeta
from utils import statistic_data
from handler.funcs import *
import copy


class AgentHandler:
    __metaclass__ = ABCMeta

    def __init__(self, name, full_game=False):
        self.name = name
        self.full_game = full_game
        self.interface = None
        self.init_episode()

    def init_episode(self):
        self.pre_state = None
        self.pre_result_list = []
        self.pre_result_list_global = []
        self.last_action = None

    def should_fixed(self, packet):
        txt = packet.get("text", "")
        if txt == "all_fixed":
            return True
        return False

    def handle_info_packet(self, json_packet, scene):
        """
        For generating the experience during normal interaction with environment.
        :param json_packet: raw game states from game clients
        :param scene:
        :return: action
        """
        my_index = json_packet['my_index']
        # get available actions to mask unavailable actions
        available_actions = self.get_available_actions(json_packet, scene, ending_object=None) if not self.full_game \
            else self.get_available_actions_share(json_packet, scene, ending_object=None)
        # get the global observations
        result_list_global = self.feature_extract_global(json_packet)
        # get the local observations
        result_list = self.feature_extract_local(json_packet, scene, my_index) if not self.full_game else \
            self.feature_extract_local_share(json_packet, scene, my_index)
        # get the middle reward
        reward = self.calc_middle_reward(json_packet) if self.pre_state and "last_action" in json_packet else 0
        # send the generated experience to corresponding interface for further collection by the algorithm
        self.interface.SendSample(my_index, scene, self.pre_result_list_global, self.pre_result_list, self.last_action,
                                  reward, result_list_global, result_list, available_actions, False,
                                  no_train=self.should_fixed(json_packet))
        # some resets
        self.last_action = self.interface.ReceiveAction()
        self.pre_state = json_packet
        self.pre_result_list = result_list
        self.pre_result_list_global = result_list_global
        # return the action
        return self.last_action

    def handle_ending_result(self, ending_object, now_state, scene=None):
        """
        For generating the last experience when episode done.
        :param ending_object: info contained in raw game states
        :param now_state: current raw game states
        :param scene:
        """
        game_over_str = ending_object["result"]
        state = ending_object['state']
        my_index = now_state['my_index']

        if game_over_str in ['passed', 'get_ball']:
            if scene == 'attack':  # include attack and assist
                return   # we don't regard this as an episode done.
            elif scene == 'ballclear':  # include ballclear and assist.
                ball_owner_team = now_state['ball']['team']
                ball_owner_index = now_state['ball']['owner_index']
                ball_owner_in_three_point_area = now_state['teams'][ball_owner_team][ball_owner_index]['three_point_area']
                if not ball_owner_in_three_point_area:
                    return  # we don't regard this as an episode done.
        # get the available actions to mask unavailable actions
        available_actions = self.get_available_actions(now_state, scene, ending_object=ending_object) if not \
            self.full_game else self.get_available_actions_share(now_state, scene, ending_object=ending_object)
        # get the final reward
        reward = self.calc_final_reward(ending_object, now_state)
        if self.pre_result_list:
            self.interface.SendSample(my_index, scene, self.pre_result_list_global, self.pre_result_list, self.last_action, reward,
                                      self.pre_result_list_global, self.pre_result_list, available_actions, True,
                                      no_train=self.should_fixed(now_state))
        # initiate the episode settings
        self.init_episode()

        # add statistical data for evaluation along with training
        now_state['game_state']['realtime'] = round(now_state['game_state']['realtime'], 2)
        if now_state.get("text", "") == "all_fixed":
            self.statistic_message(now_state)

    def statistic_message(self, data):
        uuid = data['game_state']['uuid']
        if uuid not in statistic_data.game_data:
            statistic_data.game_data[uuid] = []
            statistic_data.statistic_list[uuid] = []
        if len(statistic_data.game_data[uuid]) == 0 or data['game_state']['realtime'] > \
                statistic_data.game_data[uuid][0][-1]:
            for i in range(len(data['teams'])):
                for j in range(len(data['teams'][i])):
                    temp = data['teams'][i][j]['CharacterResult'][1:]
                    temp.append(data['game_state']['realtime'])
                    if len(statistic_data.game_data[uuid]) < len(data['teams']) * len(data['teams'][i]):
                        statistic_data.game_data[uuid].append(temp)
                    else:
                        statistic_data.game_data[uuid][i * len(data['teams'][i]) + j] = temp
            statistic_data.statistic_list[uuid].append(copy.deepcopy(statistic_data.game_data[uuid]))
            for i in range(len(statistic_data.statistic_list[uuid])):
                if statistic_data.statistic_list[uuid][i][0][-1] > statistic_data.statistic_list[uuid][-1][0][-1] - \
                        statistic_data.statistic_game_time:
                    break
            if i >= 2:
                for j in range(i - 1):
                    del statistic_data.statistic_list[uuid][0]
            statistic_data.statistic_plus[uuid] = []
            for i in range(len(statistic_data.game_data[uuid])):
                temp = [statistic_data.statistic_list[uuid][-1][i][0],
                        statistic_data.statistic_list[uuid][-1][i][1]]
                for j in range(len(statistic_data.game_data[uuid][i]) - 3):
                    temp.append(statistic_data.statistic_list[uuid][-1][i][j + 2] -
                                statistic_data.statistic_list[uuid][0][i][j + 2])
                temp.append(statistic_data.statistic_list[uuid][-1][i][-1])
                statistic_data.statistic_plus[uuid].append(temp)
            statistic_data.my_team_data[uuid] = data['my_team']

    @abstractmethod
    def calc_middle_reward(self, now_state):
        raise Exception("Not Implement " + self.name)

    @abstractmethod
    def calc_final_reward(self, ending_object, now_state):
        raise Exception("Not Implement " + self.name)

    @abstractmethod
    def feature_extract_global(self, now):
        # player info:
        global_state = []
        for team in now['teams']:
            for i in range(len(team)):
                global_state.append(team[i]["position"]["x"] / 7.5)
                global_state.append(team[i]["position"]["z"] / 13.78)
                global_state.append(team[i]['basket_distance'] / 12)
                global_state.append(team[i]["give_me_the_ball"])
                global_state.append(team[i]["shoot_rate"] / 100)
                global_state.append(float(team[i]["cannot_dribble"]))
                global_state.append(float(team[i]["three_point_area"]))
                global_state.append(float(team[i]["can_steal"]))
        # ball info:
        global_state.append(now["ball"]["position"]["x"] / 7.5)
        global_state.append(now["ball"]["position"]["y"] / 3.8)
        global_state.append(now["ball"]["position"]["z"] / 13.78)
        global_state.append(distance_of_vector3(now["ball"]["position"], RIM_POINT) / 18)
        # game info:
        global_state.append(now["game_state"]["attack_remain_time"] / 20)
        global_state.extend(get_game_state_one_hot(now["game_state"]))
        return global_state

    @abstractmethod
    def feature_extract_local(self, now, scene, index):
        raise Exception("Not Implement " + self.name)

    @abstractmethod
    def feature_extract_local_share(self, now, scene, index):
        # player info:
        local_state = []
        # add player index info
        index_onehot = [0] * 3
        index_onehot[index] = 1
        local_state.extend(index_onehot)
        # add scene info
        state = now['state']
        state_onehot = [0] * 5
        state_list = ['attack', 'assist', 'defense', 'freeball', 'ballclear']
        state_index = [i for i, s in enumerate(state_list) if s == state][0]
        state_onehot[state_index] = 1
        local_state.extend(state_onehot)

        for team in now['teams']:
            for i in range(len(team)):
                local_state.append(team[i]["position"]["x"] / 7.5)
                local_state.append(team[i]["position"]["z"] / 13.78)
                local_state.append(team[i]['basket_distance'] / 12)
                local_state.append(team[i]["give_me_the_ball"])
                local_state.append(team[i]["shoot_rate"] / 100)
                local_state.append(float(team[i]["cannot_dribble"]))
                local_state.append(float(team[i]["three_point_area"]))
                local_state.append(float(team[i]["can_steal"]))
        # ball info:
        local_state.append(now["ball"]["position"]["x"] / 7.5)
        local_state.append(now["ball"]["position"]["y"] / 3.8)
        local_state.append(now["ball"]["position"]["z"] / 13.78)
        local_state.append(distance_of_vector3(now["ball"]["position"], RIM_POINT) / 18)
        # game info:
        local_state.append(now["game_state"]["attack_remain_time"] / 20)
        local_state.extend(get_game_state_one_hot(now["game_state"]))
        return local_state

    @abstractmethod
    def get_available_actions(self, now, scene, ending_object):
        raise Exception("Not Implement " + self.name)

    @abstractmethod
    def get_available_actions_share(self, now, scene, ending_object):
        raise Exception("Not Implement " + self.name)

