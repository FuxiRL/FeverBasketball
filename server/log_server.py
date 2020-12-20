#coding=utf-8
"""
The log server for test.
"""

import json
import cgi
import argparse
import threading
import os
import tensorflow as tf
import urllib.request
import socket
import sys
import time
import copy
from utils import statistic_data
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

localhost = "127.0.0.1"


def startLocalServer(ip=localhost, port=5000):
    flag = False
    if isUsed(ip, port):
        print("%s:%d is used" % (ip, port))
        for i in range(port+1, port+100):
            if not isUsed(ip, i):
                port = i
                print("port changes to %d" % port)
                flag = True
                break
        if not flag:
            print("no port is idle, exit")
            sys.exit()

    print("server is starting at %s:%d" % (ip, port))
    ms = monitor_server(port=port)
    ms.start()
    # client
    cr = client_request(port=port)
    return cr


def isUsed(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    res = True
    try:
        s.connect((ip, port))
        s.shutdown(2)
        res = True
    except:
        res = False
    return res


class client_request():
    def __init__(self, port, ip=localhost):
        self.ip = ip
        self.port = port

    def http_post(self, data):
        url = "http://" + self.ip + ":" + str(self.port)
        values = data
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        jdata = json.dumps(values)
        req = urllib.request.Request(url, jdata, headers)
        response = urllib.request.urlopen(req)
        return response.read()


class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass


class monitor_server(threading.Thread):
    def __init__(self, port, ip=localhost):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port

    def run(self):
        http_server = ThreadingHttpServer((self.ip, self.port), HttpHandler)
        http_server.serve_forever()


monitor = {}
handler_lock = threading.Lock()

plot_lock = threading.Lock()
steps = {}
agent_step = {}
summary_writers = {}
log_dir = "./data/logs"
if not os.path.exists(log_dir): os.makedirs(log_dir)
lzh_summary_writers = {'score0': tf.summary.FileWriter(os.path.join(log_dir, 'team0')), 'score1': tf.summary.FileWriter(os.path.join(log_dir, 'team1'))}
tag = ['2points', '2points_try', '3points', '3points_try', 'block', 'steal', 'intercept', 'assist', 'rebound', 'score', 'highlight_turn', 'dribble', 'loose', 'pass', 'dunk']
position_C = [4, 61, 116, 21, 30, 52, 80, 63, 66, 81, 111, 64, 67, 95, 33, 90, 88, 119]
position_PF = [13, 75, 42, 58, 37, 51, 62, 118, 79, 96, 28, 68, 69, 70, 115, 101, 102, 113, 123, 117, 122]
position_SF = [2, 20, 73, 78, 110, 91, 98, 53, 59, 85, 14, 50, 76, 45, 46, 84, 104, 105, 106, 107, 108, 12, 24, 77, 121, 124]
position_PG = [5, 86, 27, 128, 11, 49, 97, 56, 112, 71, 99, 89, 83, 109, 120, 125]
position_SG = [57, 60, 72, 44, 92, 65, 22, 17, 18, 23, 8, 25, 82, 93, 127, 39, 74, 48, 35, 47, 87, 94, 100, 103, 114, 126]
position_list = ['5', '13', '14', '18', '33']
start_time = time.time()
statistic_data.real_time_start = start_time


def position(myid):
    if myid in position_C:
        return '18'
    if myid in position_PF:
        return '13'
    if myid in position_SF:
        return '14'
    if myid in position_PG:
        return '5'
    if myid in position_SG:
        return '33'
    return False


def tensorboard_summary_lzh(data_dict, team_dict, update_dict):
    if len(data_dict) > 0:
        client_count = len(data_dict)
        statistic_data.time_min += 1
        total_score = [0, 0]
        client_score_list = [[0 for _ in range(client_count)], [0 for _ in range(client_count)]]
        num = 0
        # scores for both team
        for client_num, key in enumerate(data_dict):
            data = data_dict[key]
            if update_dict[key]:
                num += 1
                for j in range(len(data)):
                    if key in team_dict and update_dict[key]:
                        if data[j][0] == team_dict[key]:
                            total_score[0] += data[j][11]
                            client_score_list[0][client_num-1] += data[j][11]
                        else:
                            total_score[1] += data[j][11]
                            client_score_list[1][client_num-1] += data[j][11]
        summary0 = tf.Summary()
        summary1 = tf.Summary()
        if num > 0:
            summary0.value.add(tag='team_score', simple_value=total_score[0] / num)
            summary1.value.add(tag='team_score', simple_value=total_score[1] / num)
            with open('./data/logs/score.log', 'a') as f:
                f.write(json.dumps(client_score_list))
                f.write('\n')
        else:
            summary0.value.add(tag='team_score', simple_value=0)
            summary1.value.add(tag='team_score', simple_value=0)
            # with open('./logs/score.log', 'a') as f:
            #     f.write(str(0))
            #     f.write(',')
            #     f.write(str(0))
            #     f.write('\n')

        lzh_summary_writers['score0'].add_summary(summary0, statistic_data.time_min)
        lzh_summary_writers['score1'].add_summary(summary1, statistic_data.time_min)


def tensorboard_summary(log):
    agent_id = log["agent_id"]
    plot_tag = agent_id+"_"+log["data2est"]  # "attack_loss"  "attack_reward"
    # plot_id = agent_id+"_"+log["data2est"]  # "attack_loss"  "attack_reward"
    if plot_tag not in agent_step:
        agent_step[plot_tag] = 1
    ptype = log["plot_type"]
    plot_id = plot_tag + ptype
    for i in range(len(log[ptype])):
        if plot_id not in summary_writers:
            # summary_writers[plot_id] = tf.summary.FileWriter(os.path.join(log_dir, plot_tag, ptype))
            summary_writers[plot_id] = tf.summary.FileWriter(os.path.join(log_dir, 'episode_r'))
        summary = tf.Summary(value=[tf.Summary.Value(tag=plot_tag, simple_value=log[ptype][i])])
        summary_writers[plot_id].add_summary(summary, agent_step[plot_tag])
        agent_step[plot_tag] += 1

agent_lock = {}


class HttpHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        ctype, pdict = cgi.parse_header(self.headers['content-type'])

        if ctype != 'application/json':
            self.send_error(415, "Only json data is supported.")
            return

        length = int(self.headers['content-length'])
        logs = json.loads(self.rfile.read(length))  # post_value

        # self.send_response(200)
        self.send_response_only(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"accept_list": []}).encode())
        plot_log = {"data2est": logs["data2est"],
                    "agent_id": logs["agent_id"],
                    "plot_type": logs["data2est"],
                    logs["data2est"]: logs[logs["data2est"]],
                    }
        for key in logs['lzh']:
            if (key not in statistic_data.data_dict or logs['lzh'][key][0][-1] > statistic_data.data_dict[key][0][-1]) and logs['lzh'][key][0][12] >= 0:
                statistic_data.data_dict[key] = logs['lzh'][key]
                statistic_data.update_dict[key] = True
        for key in logs['lzh_team']:
            statistic_data.team_dict[key] = logs['lzh_team'][key]

        now_time = time.time()
        statistic_data.lock.acquire()
        if now_time > statistic_data.real_time_start + statistic_data.statistic_real_time:
            tensorboard_summary_lzh(statistic_data.data_dict, statistic_data.team_dict, statistic_data.update_dict)
            for key in statistic_data.update_dict:
                statistic_data.update_dict[key] = False
            statistic_data.real_time_start = copy.deepcopy(now_time)
        statistic_data.lock.release()

    def do_GET(self):
        buf = "it works"
        self.protocal_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Welcome", "Contect")
        self.end_headers()
        self.wfile.write(buf.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch servers')
    parser.add_argument('-i', '--ip', type=str, help='ip', default=localhost)
    parser.add_argument('-p', '--port', type=int, help='port', default=6099)
    args, unknown = parser.parse_known_args()

    server_ip = args.ip
    server_port = args.port
    flag = False
    if isUsed(server_ip, server_port):
        print("%s:%d is used" % (server_ip, server_port))
        for i in range(server_port+1, server_port+100):
            if not isUsed(server_ip, i):
                server_port = i
                print("port changes to %d" % server_port)
                flag = True
                break
        if not flag:
            print("no port is idle, exit")
            sys.exit()

    print("log server is starting at %s:%d" % (server_ip, server_port))
    ms = monitor_server(port=server_port, ip=server_ip)
    ms.start()



