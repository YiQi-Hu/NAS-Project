import random
import time
import os
import copy
import socket
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager

from .enumerater import Enumerater
from .evaluator import Evaluator
# from .optimizer import Optimizer
# from .sampler import Sampler

NETWORK_POOL_TEMPLATE = []
NETWORK_POOL = []
gpu_list = multiprocessing.Queue()

if "1.txt" in os.listdir('./'):
	os.remove("1.txt")


def call_eva(graph, cell, nn_preblock, round, pos, finetune_signal, pool_len, eva, ngpu):
	print("network: {} eva pid: {} gpu: {}".format(pos, os.getpid(), ngpu))
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu)
	with open("memory/evaluating_log_with_gpu{}.txt".format(ngpu), "a") as f:
		f.write("\nblock_num:{} round:{} network_index:{}/{}".format(len(nn_preblock)+1, round, pos, pool_len))
		start_time = time.time()
		# while True:
		# 	try:
		score = eva.evaluate(graph, cell, nn_preblock, False, finetune_signal, f)
			# 	break
			# except:
			# 	print("\nevaluating failed and we will try again...\n")
			# 	f.write("\nevaluating failed and we will try again...\n")

		end_time = time.time()
	time_cost = end_time - start_time
	gpu_list.put(ngpu)
	return score, time_cost, pos


def initialize_ops_subprocess(NETWORK_POOL):
	from .predictor import Predictor
	pred = Predictor()
	print("init ops pid:", os.getpid())
	for network in NETWORK_POOL:  # initialize the full network by adding the skipping and ops to graph_part
		cell, graph = network.spl.sample()
		# network.graph_full_list.append(graph)
		blocks = []
		for block in network.pre_block:  # get the graph_full adjacency list in the previous blocks
			blocks.append(block[1])  # only get the graph_full in the pre_bock

		# process the ops from predictor
		pred_ops = pred.predictor(blocks, graph)
		table = network.spl.init_p(pred_ops)  # spl refer to the pred_ops
		network.spl.renewp(table)
		cell, graph = network.spl.convert()  # convert the table

		# graph from first sample and second sample are the same, so that we don't have to assign network.graph_full at first time
		network.graph_full_list.append(graph)
		network.cell_list.append(cell)
	return NETWORK_POOL


def train_winner_subprocess(eva, NETWORK_POOL, add_data_for_winner, opt_best_k):
	best_nn = NETWORK_POOL[0]
	best_opt_score = 0
	best_cell_i = 0
	eva.add_data(add_data_for_winner)  # -1 represent that we add all data for training
	print("NAS: Configuring ops and skipping for the best structure and training them...")
	for i in range(opt_best_k):
		cell, graph = best_nn.spl.sample()
		best_nn.graph_full_list.append(graph)
		best_nn.cell_list.append(cell)
		with open("memory/train_winner_log.txt", "a") as f:
			f.write("\nblock_num:{} sample_count:{}/{}".format(len(best_nn.pre_block) + 1, i, opt_best_k))
			opt_score = eva.evaluate(graph, cell, best_nn.pre_block, True, True, f)
		best_nn.score_list.append(opt_score)
		if opt_score > best_opt_score:
			best_opt_score = opt_score
			best_cell_i = i
	print("NAS: We have got the best network and its score is {}".format(best_opt_score))
	best_index = best_cell_i - opt_best_k
	return best_nn, best_index


def run_global(eva,graph,cell,ngpu,gpu_list):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu);k=0
	#while k<5:
		#try:
	score=eva.evaluate(graph,cell,[]);#break;
		#except:
			#k+=1;score=0.1
	print(str(ngpu)+"  achieve",cell,graph)
	gpu_list.put(ngpu)
	return score


class QueueManager(BaseManager):
	pass


class Communication:
	def __init__(self, role, ps_host):
		task_queue = multiprocessing.Queue()
		result_queue = multiprocessing.Queue()
		flag = multiprocessing.Queue()
		data_queue = multiprocessing.Queue()

		QueueManager.register('get_task_queue', callable=lambda: task_queue)
		QueueManager.register('get_result_queue', callable=lambda: result_queue)
		QueueManager.register('get_flag', callable=lambda: flag)
		QueueManager.register('get_data_sync', callable=lambda: data_queue)

		# TODO there might be other ways to get the IP address
		server_addr = socket.gethostbyname(ps_host.split(":")[0])
		self.manager = QueueManager(address=(server_addr, int(ps_host.split(":")[1])), authkey=b'abc')

		if role == "ps":
			self.manager.start()
		else:
			while True:
				try:
					self.manager.connect()
					break
				except:
					time.sleep(20)
					print("waiting for connecting ...")

		self.task = self.manager.get_task_queue()
		self.result = self.manager.get_result_queue()
		self.end_flag = self.manager.get_flag()  # flag for whether the whole process is over
		self.data_sync = self.manager.get_data_sync()  # flag for sync of adding data and sync of round
		self.data_count = 0  # mark how many times to add data locally

	def over(self):
		self.end_flag.put(1)
		self.manager.shutdown()


class Nas:
	def __init__(self, pattern, setting, search_space):
		self.__pattern = pattern

		self.__m_best = setting["m_best"]
		self.__opt_best_k = setting["opt_best_k"]
		self.num_gpu = setting["num_gpu"]
		self.ps_host = setting["ps_host"]
		self.worker_host = setting["worker_host"]
		self.job_name = setting["job_name"]
		self.task_index = setting["task_index"]
		self.__m_pool = []
		self.__finetune_threshold = setting["finetune_threshold"]

		self.__block_num = search_space["block_num"]
		if self.job_name == "ps":
			print("NAS: Initializing enu...")
			self.enu = Enumerater(search_space["graph"])
		print("NAS: Initializing eva...")
		self.eva_para = setting["eva_para"]
		self.eva = Evaluator(self.eva_para)
		print("NAS: Initializing com...")
		self.com = Communication(self.job_name, self.ps_host)

		for gpu in range(self.num_gpu):
			gpu_list.put(gpu)

		self.spl_setting = setting["spl_para"]
		self.skipping_max_dist = search_space["skipping_max_dist"]
		self.ops_space = search_space["ops"]

		self.add_data_every_round = setting["add_data_every_round"]
		self.add_data_for_winner = setting["add_data_for_winner"]

		if setting["rand_seed"] is not -1:
			random.seed(setting["rand_seed"])
			tf.set_random_seed(setting["rand_seed"])

		return



	def __list_swap(self, ls, i, j):
		cpy = ls[i]
		ls[i] = ls[j]
		ls[j] = cpy

	def __eliminate(self, network_pool=None, round=0):
		"""
		Eliminates the worst 50% networks in network_pool depending on scores.
		"""
		scores = [network_pool[nn_index].score_list[-1] for nn_index in range(len(network_pool))]
		print(scores)
		scores.sort()
		original_num = len(scores)
		mid_index = original_num // 2
		mid_val = scores[mid_index]
		print(mid_val)
		original_index = [i for i in range(len(scores))]  # record the number of the removed network in the pool

		i = 0
		while i < len(network_pool):
			if network_pool[i].score_list[-1] < mid_val:
				self.__list_swap(network_pool, i, len(network_pool) - 1)
				self.__list_swap(original_index, i, len(original_index) - 1)
				self.save_info("memory/network_info.txt", network_pool.pop(), round, original_index.pop(), original_num)
			else:
				i += 1
		print("NAS: eliminating {}, remaining {}...".format(original_num - len(network_pool), len(network_pool)))
		return mid_val

	def save_info(self, path, network, round, original_index, network_num):
		with open(path, 'a') as f:
			f.write("block_num: {} round: {} network_index: {}/{} number of scheme: {}\n".format(len(network.pre_block)+1, round, original_index, network_num, len(network.score_list)))
			f.write("graph_part:")
			self.wirte_list(f, network.graph_part)
			for item in zip(network.graph_full_list, network.cell_list, network.score_list):
				f.write("    graph_full:")
				self.wirte_list(f, item[0])
				f.write("    cell_list:")
				self.wirte_list(f, item[1])
				f.write("    score:")
				f.write(str(item[2]) + "\n")

	def wirte_list(self, f, graph):
		f.write("[")
		for node in graph:
			f.write("[")
			for ajaceny in node:
				f.write(str(ajaceny) + ",")
			f.write("],")
		f.write("]" + "\n")

	def __game(self, eva, NETWORK_POOL, com, round):
		pool_len = len(NETWORK_POOL)
		print("NAS: Now we have {0} networks. Start game!".format(pool_len))
		network_index = 0
		# put all the network in this round into the task queue
		if pool_len < self.__finetune_threshold:
			finetune_signal = True
		else:
			finetune_signal = False
		for nn in NETWORK_POOL:
			if round == 1:
				cell, graph = nn.cell_list[-1], nn.graph_full_list[-1]
				# cell, graph = nn.spl.sample()
			else:
				nn.spl.update_opt_model(nn.score_list[-1])
				cell, graph = nn.spl.sample()
				nn.graph_full_list.append(graph)
				nn.cell_list.append(cell)
			com.task.put([graph, cell, nn.pre_block, network_index, round, finetune_signal, pool_len])
			network_index += 1

		# TODO data size control
		com.data_sync.put(com.data_count)  # for multi host
		eva.add_data(self.add_data_every_round)

		pool = Pool(self.num_gpu)
		# as long as there's still task available
		eva_result_list = []
		while not com.task.empty():
			gpu = gpu_list.get()  # get a gpu resource
			try:
				graph, cell, nn_pre_block, network_index, round, finetune_signal, pool_len = com.task.get(timeout=1)
			except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
				gpu_list.put(gpu)
				break
			print("round:{} network_index:{}".format(round, network_index))
			print("graph:", graph)
			print("cell:", cell)
			eva_result = pool.apply_async(call_eva, args=(graph, cell, nn_pre_block, round, network_index, finetune_signal, pool_len, eva, gpu))
			eva_result_list.append(eva_result)
		pool.close()
		pool.join()

		for eva_result in eva_result_list:
			score, time_cost, network_index = eva_result.get()
			print("network_index:{} score:{} time_cost:{} ".format(network_index, score, time_cost))
			com.result.put([network_index, score, time_cost])

		while com.result.qsize() != len(NETWORK_POOL):  # waiting for the other workers
			print("we have gotten {} scores , but there are {} networks, waiting for the other workers...".format(com.result.qsize(), len(NETWORK_POOL)))
			time.sleep(20)
		# fill the score list
		print("network_index score time_cost")
		while not com.result.empty():
			network_index, score, time_cost = com.result.get()
			print([network_index, score, time_cost])
			NETWORK_POOL[network_index].score_list.append(score)

	def __train_winner(self, NETWORK_POOL, round):
		eva_winner = Evaluator(self.eva_para)
		with Pool(1) as p:
			best_nn, best_index = p.apply(train_winner_subprocess, args=(eva_winner, NETWORK_POOL, self.add_data_for_winner, self.__opt_best_k))
		self.save_info("memory/network_info.txt", best_nn, round, 0, 1)
		return best_nn, best_index

	def initialize_ops(self, NETWORK_POOL):
		print("NAS: Configuring the networks in the first round...")
		with Pool(1) as p:
			NETWORK_POOL = p.apply(initialize_ops_subprocess, args=(NETWORK_POOL,))
		return NETWORK_POOL

	def algorithm_ps(self, block_num, eva, com, NETWORK_POOL_TEMPLATE):
		# implement the copy when searching for every block
		NETWORK_POOL = copy.deepcopy(NETWORK_POOL_TEMPLATE)
		for network in NETWORK_POOL:  # initialize the sample module
			network.init_sample(self.__pattern, block_num, self.spl_setting, self.skipping_max_dist, self.ops_space)

		NETWORK_POOL = self.initialize_ops(NETWORK_POOL)
		round = 0
		while (len(NETWORK_POOL) > 1):
			# Step 3: Sample, train and evaluate every network
			com.data_count += 1
			round += 1
			self.__game(eva, NETWORK_POOL, com, round)
			# Step 4: Eliminate half structures and increase dataset size
			self.__eliminate(NETWORK_POOL, round)
			# self.__datasize_ctrl("same", epic)
		print("NAS: We got a WINNER!")
		# Step 5: Global optimize the best network
		best_nn, best_index = self.__train_winner(NETWORK_POOL, round+1)
		# self.__save_log("", opt, spl, enu, eva)
		return best_nn, best_index

	def algorithm_worker(self, eva, com):
		while com.end_flag.empty():
			# as long as there's still task available
			# Data control for new round
			while com.data_sync.empty():
				print("waiting for assignment of next round...")
				time.sleep(20)
			com.data_count += 1
			data_count_ps = com.data_sync.get(timeout=1)
			assert com.data_count <= data_count_ps, "add data sync failed..."
			eva.add_data(self.add_data_every_round*(data_count_ps-com.data_count+1))  # 1600
			pool = Pool(self.num_gpu)
			eva_result_list = []
			while not com.task.empty():
				gpu = gpu_list.get()  # get a gpu resource
				try:
					graph, cell, nn_pre_block, network_index, round, finetune_signal, pool_len = com.task.get(timeout=1)
				except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
					gpu_list.put(gpu)
					break
				print("round:{} network_index:{}".format(round, network_index))
				print("graph:", graph)
				print("cell:", cell)
				eva_result = pool.apply_async(call_eva, args=(graph, cell, nn_pre_block, round, network_index, finetune_signal, pool_len, eva, gpu))
				eva_result_list.append(eva_result)
			pool.close()
			pool.join()
			for eva_result in eva_result_list:
				score, time_cost, network_index = eva_result.get()
				print("network_index:{} score:{} time_cost:{} ".format(network_index, score, time_cost))
				com.result.put([network_index, score, time_cost])
			while com.task.empty():
				print("waiting for assignment of next round...")
				time.sleep(20)

		return "I am a worker..."

	def dis_global(self):
		manager = multiprocessing.Manager()
		gpu_list = manager.Queue()
		f1=open("1.txt",'a')
		# Step 1: Brute Enumerate all possible network structures and initialize spl and opt for every network
		print("NAS: Enumerating all possible networks...")
		NETWORK_POOL = self.enu.enumerate()
		block_num=0
		for network in NETWORK_POOL:
			network.init_sample(self.__pattern, block_num, self.spl_setting, self.skipping_max_dist, self.ops_space)
			# network.init_sample(self.__pattern, block_num)
		with Pool(1) as p:
			NETWORK_POOL=p.apply(initialize_ops_subprocess,(NETWORK_POOL,));
		nn=NETWORK_POOL[0];
		cells=[];graphs=[];pros=[];
		print(nn.cell_list[0],nn.graph_full_list[0])
		cells.append(nn.cell_list[0]);graphs.append(nn.graph_full_list[0]);gpu_list.put(0)
		pros.append(nn.spl.opt.sample());nn.spl.renewp(pros[0])
		for gpu in range(1,self.num_gpu):
			gpu_list.put(gpu)
			cell, graph = nn.spl.convert();cells.append(cell);graphs.append(graph)
			pros.append( nn.spl.opt.sample())
			nn.spl.renewp(pros[gpu])
		print(cells,graphs)
		pool= Pool(processes=self.num_gpu);self.eva.add_data(self.add_data_for_winner);results=[];i=0;
		while i<200:
			while not gpu_list.empty():
				ngpu=gpu_list.get();#eva.add_data(800)
				#print(gpu_list.qsize())
				eva_result=pool.apply_async(run_global,args=(self.eva,graphs[i],cells[i],ngpu,gpu_list,))
				#print(eva_result,i,ngpu,cells[i],graphs[i],flush=True);
				results.append(eva_result);i=i+1;
			k=0
			for result in results[i-self.num_gpu:]:
				score=result.get();
				print("score",score)
				while True:
					try:
						nn.opt.update_model(pros[k],score);break;
					except:
						print('we sampling again!')
				pros[k]= nn.spl.opt.sample()
				nn.spl.renewp(pros[k])
				cell, graph = nn.spl.convert();cells.append(cell);graphs.append(graph);s=''
				for x in cells[i-self.num_gpu+k]:
					x=str(x)
					st=str(x.split(','))
					for x in st:
						s=s+x
				for x in graphs[i-self.num_gpu+k]:
					x=str(x)
					st=str(x.split(','))
					for x in st:
						s=s+x
				s=s+'\n'
				f1.write(str(score)+' '+s);
				f1.flush()
				k=k+1;
		pool.close()#
		pool.join()#

	def run(self):
		if self.__pattern == "Global":
			self.dis_global()
			return "we test the Global NAS"
		if self.job_name == "ps":
			print("NAS: Enumerating all possible networks!")
			NETWORK_POOL_TEMPLATE = self.enu.enumerate()
			for i in range(self.__block_num):
				print("NAS: Searching for block {}/{}...".format(i + 1, self.__block_num))
				block, best_index = self.algorithm_ps(i, self.eva, self.com, NETWORK_POOL_TEMPLATE)
				block.pre_block.append([block.graph_part, block.graph_full_list[best_index], block.cell_list[best_index]])
				# or NetworkUnit.pre_block.append()
			self.com.over()
			return block.pre_block  # or NetworkUnit.pre_block
		else:
			self.algorithm_worker(self.eva, self.com)
			return "all of the blocks have been evaluated, please go to the ps manager to view the result..."


if __name__ == '__main__':
	nas = Nas()
	print(nas.run())
