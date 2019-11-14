import os
import json

# Current file name
CUR_VER_DIR = os.getcwd()

# NAS configuration dic object
_NAS_CONFIG_PATH = os.path.join(CUR_VER_DIR, 'nas_config.json')
NAS_CONFIG = json.load(open(_NAS_CONFIG_PATH, encoding='utf-8'))

# File path
if not os.path.exists("memory"):
    os.mkdir("memory")
LOG_DIR = os.path.join(CUR_VER_DIR, 'memory')
EVALOG_PATH_TEM = os.path.join(LOG_DIR, 'evaluating_log_with_gpu{}.txt')
NETWORK_INFO_PATH = os.path.join(LOG_DIR, 'network_info.txt')
WINNER_LOG_PATH = os.path.join(LOG_DIR, 'train_winner_log.txt')

# Log content
LOG_EVAINFO_TEM = '\nblock_num:{} round:{} network_index:{}/{} spl_index: {}/{} eva_pid: {} gpu: {}\n'
LOG_EVAFAIL = 'evaluating failed and we will try again...\n'
LOG_WINNER_TEM = 'block_num:{} sample_count:{}/{}\n'

# System information
SYS_EVAFAIL = 'NAS: ' + LOG_EVAFAIL
SYS_EVA_RESULT_TEM = 'NAS: network_index:{} spl_index:{} score:{} time_cost:{} '
SYS_ELIINFO_TEM = 'NAS: eliminating {}, remaining {}...'
SYS_INIT_ING = 'NAS: Initializing...'
SYS_I_AM_PS = 'NAS: I am ps.'
SYS_I_AM_WORKER = 'NAS: I am worker.'
SYS_ENUM_ING = 'NAS: Enumerating all possible networks!'
SYS_SEARCH_BLOCK_TEM = 'NAS: Searching for block {}/{}...'
SYS_WORKER_DONE = 'NAS: all of the blocks have been evaluated, please go to the ps manager to view the result...'
SYS_WAIT_FOR_TASK = 'NAS: waiting for assignment of next round...'
SYS_CONFIG_ING = 'NAS: Configuring the networks in the first round...'
SYS_ROUND_OVER = "NAS: The round is over, cost time: {}"
SYS_GET_WINNER = "NAS: We got a WINNER and game cost: {}"
SYS_BEST_AND_SCORE_TEM = 'NAS: We have got the best network and its score is {}'
SYS_START_GAME_TEM = 'NAS: Now we have {} networks. Start game!'
SYS_CONFIG_OPS_ING = "NAS: Configuring ops and skipping for the best structure and training them..."
SYS_SEARCH_FIN = "NAS: Search finished and cost time: {}"
SYS_CUR_BLK_SEARCH_FIN = "NAS: Search current block finished and cost time: {}"
SYS_TRAIN_WINNER_FIN = "NAS: Train winner finished and cost time: {}"
