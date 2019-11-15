import os
import json

# Current file name
_cur_ver_dir = os.getcwd()

# NAS configuration dic object
_nas_config_path = os.path.join(_cur_ver_dir, 'nas_config.json')

NAS_CONFIG = json.load(open(_nas_config_path, encoding='utf-8'))

# TODO Move to logger.py
# File path
# evalog_path_tem = os.path.join(cur_ver_dir, 'memory',
#                                 'evaluating_log_with_gpu{}.txt')
# network_info_path = os.path.join(cur_ver_dir, 'memory', 'network_info.txt')
# winner_log_path = os.path.join(cur_ver_dir, 'memory', 'train_winner_log.txt')

# Log content
# log_evainfo_tem = 'block_num:{} round:{} network_index:{}/{}\n'
# log_evafail = 'evaluating failed and we will try again...\n'
# log_winner_tem = 'block_num:{} sample_count:{}/{}\n'

# System information
evafail = 'NAS: evaluating failed and we will try again...'
eva_result_tem = 'NAS: network_index:{} score:{} time_cost:{} '
eliinfo_tem = 'NAS: eliminating {}, remaining {}...'
init_ing = 'NAS: Initializing...'
i_am_ps = 'NAS: I am ps.'
i_am_worker = 'NAS: I am worker.'
enum_ing = 'NAS: Enumerating all possible networks!'
search_block_tem = 'NAS: Searching for block {}/{}...'
worker_done = 'NAS: all of the blocks have been evaluated, please go to the ps manager to view the result...'
wait_for_task = 'NAS: waiting for assignment of next round...'
config_ing = 'NAS: Configuring the networks in the first round...'
get_winner = 'NAS: We got a WINNER!'
best_and_score_tem = 'NAS: We have got the best network and its score is {}'
start_game_tem = 'NAS: Now we have {} networks. Start game!'
config_ops_ing = "NAS: Configuring ops and skipping for the best structure and training them..."
