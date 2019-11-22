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
eva_pre = 'block_num:{0[0]} round:{0[1]} network_index:{0[2]}/{0[3]} spl_index:{0[4]}/{0[5]}'
eva_result_tem = "network_index:{0[0]} spl_index:{0[1]} score:{0[2]} time_cost:{0[3]}"
eva_ing = 'block_num:{0[0]} round:{0[1]} network_index:{0[2]}/{0[3]} spl_index:{0[4]}/{0[5]}' \
          ' score:{0[6]} time_cost:{0[7]} eva_pid:{0[8]} finished...'
eliinfo_tem = 'NAS: eliminating {0[0]}, remaining {0[1]}...'
init_ing = 'NAS: Initializing...'
i_am_ps = 'NAS: I am ps.'
i_am_worker = 'NAS: I am worker.'
enum_ing = 'NAS: Enumerating all possible networks!'
search_block_tem = 'NAS: Searching for block {0[0]}/{0[1]}...'
worker_done = 'NAS: all of the blocks have been evaluated, please go to the ps manager to view the result...'
wait_for_task = 'NAS: waiting for assignment of next round...'
config_ing = 'NAS: Configuring the networks in the first round...'
get_winner = 'NAS: We got a WINNER and cost time: {0[0]}'
best_and_score_tem = 'NAS: We have got the best network and its score is {0[0]}'
start_game_tem = 'NAS: Now we have {0[0]} networks. Start game!'
config_ops_ing = "NAS: Configuring ops and skipping for the best structure and training them..."
search_fin_tem = "NAS: Search finished, cost time: {0[0]} search result:"
pre_block = "[{0[0]}, {0[1]}],"
blk_search_tem = "NAS: Search current block finished and cost time: {0[0]}"
train_winner_tem = "NAS: Train winner finished and cost time: {0[0]}"
round_over = "NAS: The round is over, cost time: {0[0]}"
retrain_end = "NAS: We retrain the final network, score {0[0]}  cost time {0[1]}"
model_save = "model{0[0]} saved..."
no_dim_spl = "There maybe no dim for sample, {0[0]} table sampled !!!"
no_dim_ini = "There maybe no dim for first sample !!!"
elim_net = "Network info of net removed\nblock_num: {0[0]} round: {0[1]} network_left: {0[2]} " \
                "network_id: {0[3]} number of scheme: {0[4]}\n"
elim_net_info = "block_num: {0[0]} round: {0[1]} network_left: {0[2]} " \
                "network_id: {0[3]} number of scheme: {0[4]}\ngraph_part:{0[5]}\n"
scheme_info = "    graph_full:{0[0]}\n    cell_list:{0[1]}\n    code:{0[2]}\n    score:{0[3]}\n"

eva = "%s"

# moudle X function X ACTION -> logger template string
MF_TEMP = {
    'nas': {
        'run': {
            'enuming': enum_ing,
            'search_blk': search_block_tem,
            'search_blk_end': blk_search_tem,
            'nas_end': search_fin_tem,
            'retrain_end': retrain_end,
            'no_dim_spl': no_dim_spl,
            'no_dim_ini': no_dim_ini,
            'pre_block': pre_block
        },
        '_arrange_result': {
            'eva_result': eva_result_tem
        },
        '_subproc_eva': {
            'eva_ing': eva_ing
        },
        'algo': {
            'start_game': start_game_tem,
            'config_ing': config_ing,
            'round_over': round_over,
            'get_winner': get_winner
        },
        '_eliminate': {
            'eliinfo_tem': eliinfo_tem,
            'elim_net': elim_net
        },
        '_save_net_info': {
            'elim_net_info': elim_net_info,
            'scheme_info': scheme_info
        },
        '_train_winner': {
            'config_ops_ing': config_ops_ing,
            'train_winner_tem': train_winner_tem
        },
        '__init__': {
            'init_ing': init_ing
        },
        '_do_task': {
            'eva_pre': eva_pre
        },
        '_rm_other_model': {
            'model_save': model_save
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
