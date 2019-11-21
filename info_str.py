import os
import json

# Current file name
_cur_ver_dir = os.getcwd()

# NAS configuration dic object
_nas_config_path = os.path.join(_cur_ver_dir, 'nas_config.json')

NAS_CONFIG = json.load(open(_nas_config_path, encoding='utf-8'))

# TODO Move to logger.py
# File path
if not os.path.exists("memory"):
    os.mkdir("memory")
log_dir = os.path.join(_cur_ver_dir, 'memory')
evalog_path = os.path.join(log_dir, 'evaluator_log.txt')
subproc_log_path = os.path.join(log_dir, 'subproc_log.txt')
network_info_path = os.path.join(log_dir, 'network_info.txt')
naslog_path = os.path.join(log_dir, 'nas_log.txt')

# Log content
# log_evainfo_tem = 'block_num:{} round:{} network_index:{}/{}\n'
# log_evafail = 'evaluating failed and we will try again...\n'
# log_winner_tem = 'block_num:{} sample_count:{}/{}\n'

# System information
evafail = 'NAS: evaluating failed and we will try again...'
eva_result_tem = 'network_index:{} score:{} time_cost:{} '
eliinfo_tem = 'NAS: eliminating {}, remaining {}...'
init_ing = 'NAS: Initializing...'
i_am_ps = 'NAS: I am ps.'
i_am_worker = 'NAS: I am worker.'
enum_ing = 'NAS: Enumerating all possible networks!'
search_block_tem = 'NAS: Searching for block %f...'
worker_done = 'NAS: all of the blocks have been evaluated, please go to the ps manager to view the result...'
wait_for_task = 'NAS: waiting for assignment of next round...'
config_ing = 'NAS: Configuring the networks in the first round...'
get_winner = 'NAS: We got a WINNER!'
best_and_score_tem = 'NAS: We have got the best network and its score is {}'
start_game_tem = 'NAS: Now we have {} networks. Start game!'
config_ops_ing = "NAS: Configuring ops and skipping for the best structure and training them..."
search_tem = "NAS: Search finished and cost time: %d"
blk_search_tem = "NAS: Search current block finished and cost time: %d"
train_winner_tem = "NAS: Train winner finished and cost time: {}"
round_over = "NAS: The round is over, cost time: {}"
retrain_end = "NAS: We retrain the final network, socre {}  cost time {}"
no_dim_spl = "There maybe no dim for sample, {} table sampled !!!"
no_dim_ini = "There maybe no dim for first sample !!!"
elim_net_info = "\nblock_num: {} round: {} network_left: {} network_id: {} number of scheme: {}\n"

eva = "%s"

# moudle X function X ACTION -> logger template string
MF_TEMP = {
    'nas': {
        'run': {
            'enuming': enum_ing,
            'search_blk': search_block_tem,
            'search_blk_end': blk_search_tem,
            'nas_end': search_tem,
            'retrain_end': retrain_end,
            'no_dim_spl': no_dim_spl,
            'no_dim_ini': no_dim_ini,
            'elim_net_info': elim_net_info
        }
    },
    'evaluator': {
        '_eval': {
            'eva': "%s"
        }
    },
    'enumerater': {
        '': ''
    },
    'predictor': {
        '': ''
    },
    'sampler': {
        '': ''
    },
    'utils': {
        '<module>': {
            'enuming': "utils: enuming",
            'hello': "uitls: %s, %s"
        }
    }
}
