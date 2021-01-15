# coding: utf-8
import urllib.request
import json
import threading
from multiprocessing import Queue
from utils import statistic_data


class LogHandler:
    def __init__(self, key, ip, port):
        self.ip = ip
        self.port = port
        self.key = key
        self.data_q = Queue()
        self.result = {}
        threading.Thread(target=self.serve).start()

    def close(self):
        self.data_q.put(None)

    def push(self, name, value):
        arr = self.result.get(name, [])
        arr.append(value)
        if len(arr) >= 1:
            process_data = {"agent_id": self.key,
                            "data2est": name, name: arr}
            self.data_q.put(process_data)
            arr = []
        self.result[name] = arr

    def serve(self):
        while True:
            item = self.data_q.get()
            if item is None:
                break
            url = "http://" + self.ip + ":" + str(self.port)
            headers = {"Content-type": "application/json", "Accept": "text/plain"}
            item['lzh'] = statistic_data.statistic_plus
            item['lzh_team'] = statistic_data.my_team_data
            jdata = json.dumps(item).encode()
            req = urllib.request.Request(url, jdata, headers)
            try:
                response = urllib.request.urlopen(req, timeout=60)
                result_str = response.read()
            except:
                pass

