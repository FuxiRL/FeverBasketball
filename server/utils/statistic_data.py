import time
import threading
statistic_data = {'time': 0, 'total_score': [0, 0], '5': [0]*15, '13': [0]*15, '14': [0]*15, '18': [0]*15, '33': [0]*15}
time_str = time.strftime("%y_%m_%d_%H_%M", time.localtime())
file_name = 'logs/lzh_log_' + time_str
statistic_tag = ['2points', '2points_try', '3points', '3points_try', 'block', 'steal', 'intercept', 'assist', 'rebound', 'score', 'team', 'highlight_turn', 'dribble', 'loose', 'pass', 'dunk']
statistic_plus = {}
statistic_list = {}
game_data = {}
statistic_real_time = 180
statistic_game_time = 180
time_min = 0
real_time_start = 0
data_dict = {}
team_dict = {}
lock = threading.Lock()
my_team_data = {}
update_dict = {}