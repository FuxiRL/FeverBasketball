# coding:utf-8
"""
The server for handling request from game clients.
"""
import threading
import socketserver
import json
import time
import traceback
import argparse
import subprocess
from handler.task_3v3.attack_agent_handler_3v3 import AttackAgentHandler3v3
from handler.task_3v3.attack_agent_handler_3v3_ma import AttackAgentHandler3v3_ma
from handler.task_3v3.defense_agent_handler_3v3 import DefenseAgentHandler3v3
from handler.task_3v3.defense_agent_handler_3v3_ma import DefenseAgentHandler3v3_ma
from handler.task_3v3.freeball_agent_handler_3v3 import FreeballAgentHandler3v3
from handler.task_3v3.freeball_agent_handler_3v3_ma import FreeballAgentHandler3v3_ma
from handler.task_3v3.ballclear_agent_handler_3v3 import BallclearAgentHandler3v3
from handler.task_3v3.ballclear_agent_handler_3v3_ma import BallclearAgentHandler3v3_ma
from handler.task_3v3.assist_agent_handler_3v3 import AssistAgentHandler3v3
from algorithm.agent import Agent
from algorithm.agent_ma import MAagent
from utils.rw_lock import RWLock
from handler.funcs import *

agent_map = {}
agent_map_lock = threading.Lock()
fixed_update_lock = RWLock("fixed_update")


class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        is_testing = args.test
        full_game = args.full
        # initialize the handlers (for generating transition experience)
        if not args.multi:
            agent_handlers = {
                "attack": AttackAgentHandler3v3("attack", full_game=full_game),
                "defense": DefenseAgentHandler3v3("defense", full_game=full_game),
                "ballclear": BallclearAgentHandler3v3("ballclear", full_game=full_game),
                "freeball": FreeballAgentHandler3v3("freeball", full_game=full_game),
                "assist": AssistAgentHandler3v3("assist", full_game=full_game)}
        else:
            agent_handlers = {
                "attack": AttackAgentHandler3v3_ma("attack", full_game=full_game),
                "defense": DefenseAgentHandler3v3_ma("defense", full_game=full_game),
                "freeball": FreeballAgentHandler3v3_ma("freeball", full_game=full_game),
                "ballclear": BallclearAgentHandler3v3_ma("ballclear", full_game=full_game)}

        # start communicate with game clients
        try:
            last_data_time = 0
            while True:
                data = self.rfile.readline()
                if data:
                    data = data.decode()
                    data = json.loads(data)

                    fixed_update_lock.reader_acquire()

                    # check which state needs to be handle
                    states = set()
                    for end_result in data.get("end_results", []):
                        states.add(end_result["state"])
                    if data["need_action"]:
                        states.add(data["state"])
                    scene = None
                    game_uuid = data["game_state"]["uuid"]
                    game_state = data["game_state"]['game_state']

                    # get the corresponding handler
                    for state in states:
                        if not args.multi:
                            handler = agent_handlers[state]
                            agent_key = data["my_type"] + "_" + state if not args.full else data["my_type"]
                        else:
                            scene = get_scene(state, game_state)
                            handler = agent_handlers[scene]
                            agent_key = data["my_type"] + "_" + scene if not args.full else data["my_type"]

                        # if the handler doesn't have the interface corresponding to the specific player yet，
                        # get it an agent to handle corresponding requests (e.g. at the start of training process).
                        if not handler.interface:
                            agent_map_lock.acquire()
                            if agent_key not in agent_map:
                                if not args.multi:
                                    state_dim = state_space_full if args.full else len(handler.feature_extract(data))
                                    action_num = action_space_full if args.full else data["action_count"][handler.name]
                                    agent_map[agent_key] = Agent(args.algorithm,
                                                                 state_dim=state_dim,
                                                                 action_num=action_num,
                                                                 model_name=agent_key,
                                                                 log_port=args.log_port,
                                                                 is_test=is_testing,)
                                else:
                                    state_dim = state_space_full if args.full else state_space_map[scene]
                                    action_num = action_space_full if args.full else action_space_map[scene]
                                    agent_map[agent_key] = MAagent(args.algorithm,
                                                                   state_dim_global=state_dim_global,
                                                                   state_dim=state_dim,
                                                                   action_num=action_num,
                                                                   model_name=agent_key,
                                                                   log_port=args.log_port,
                                                                   is_test=is_testing)
                                agent_map[agent_key].start()  # start the log thread for test
                            agent_map_lock.release()
                            handler.interface = agent_map[agent_key].add_interface(game_uuid)

                    # handle end results (when episode done), which is depended on full_game or divide_and_conquer.
                    end_results = data.get("end_results", [])
                    for end_result in end_results:
                        if not args.full or (args.full and (game_state in ['GoalIn', 'ShotclockViolation'])):
                            if not args.multi:
                                agent_handlers[end_result["state"]].handle_ending_result(end_result, data)
                            else:
                                scene = get_scene(end_result["state"], game_state)
                                agent_handlers[scene].handle_ending_result(end_result, data, scene)

                    # get the action when the corresponding player requesting action
                    if data["need_action"]:
                        if not args.multi:
                            tmp_key_map = agent_handlers[data["state"]].handle_info_packet(data) - 1
                        else:
                            scene = get_scene(data["state"], game_state)
                            tmp_key_map = agent_handlers[scene].handle_info_packet(data, scene) - 1

                    else:
                        tmp_key_map = -1
                    fixed_update_lock.reader_release()

                    # send action to game client
                    if data["need_action"]:
                        response = json.dumps([tmp_key_map])
                        self.wfile.write(response.encode())
                    last_data_time = time.clock()
                else:
                    if time.clock() - last_data_time > 1.0:
                        break
        except:
            traceback.print_exc()
        finally:
            print(threading.currentThread().name + " was not serving.")
            while True:
                time.sleep(1000)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", action='store_const', const=True, default=False,
                        help="full_game mode or divide_and_conquer mode: For full_game mode, episode only ends when a"
                             "score is made or time is up. For divide_and_conquer mode, episode ends when the sub-task"
                             "switches.")
    parser.add_argument("-m", "--multi", action='store_const', const=True, default=False,
                        help="single_agent perspective or multi_agent perspective: The difference lies in whether the"
                             "players are modeled as a team.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['dqn', 'hyq', 'exq', 'vdn', 'qmix'], default='dqn',
                        help="choose the implemented algorithm from choices.")
    parser.add_argument("-p", "--port", type=int, default=6666, help="port",)
    parser.add_argument("-l", "--log_port", help="log_port", type=int, default=6099)
    parser.add_argument("-t", "--test", action='store_const', const=True, default=False, help="is_test")
    args = parser.parse_args()

    # algorithm check
    if args.multi:
        try:
            assert args.algorithm in ['hyq', 'exq', 'vdn', 'qmix']
        except:
            raise TypeError('Please use algorithms for multi-agent training, such as [hyq, exq, vdn, qmix]')
    else:
        try:
            assert args.algorithm in ['dqn']
        except:
            raise TypeError('Please use algorithms for single-agent training, such as [dqn]')

    HOST, PORT = "0.0.0.0", args.port

    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    # start the log server
    arg_list_log = ["python", "log_server.py", "-i", "0.0.0.0", "-p", str(args.log_port)]
    subprocess.Popen(arg_list_log)

    print('Settings: full_game={} multi_agent={} algorithm={} port={} log_port={} test={}'.format(args.full,
                                                                                                  args.multi,
                                                                                                  args.algorithm,
                                                                                                  args.port,
                                                                                                  args.log_port,
                                                                                                  args.test))
    print("Server loop running in thread:", server_thread.name)
    print("Please launch game clients...")



