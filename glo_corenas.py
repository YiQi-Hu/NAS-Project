import copy
import multiprocessing
from multiprocessing import Pool
from numpy import zeros
#import nas
import random
import os,copy
#from nas import _gpu_eva
from sampler import Sampler
from base import Network ,NetworkItem ,Cell
import copy
import numpy as np
from enumerater import Enumerater
from utils import Communication, list_swap
from evaluator import Evaluator
from sampler import Sampler

from info_str import NAS_CONFIG

def rand_gen_cell(celllist,number):
	while number>0:
		flag=random.randint(0, 1)
		if flag==0:
			ra_a=random.randint(0, 6)
			conv_fiter=NAS_CONFIG['spl']['conv_space']['filter_size'][ra_a]
			ra_c,ra_d=random.randint(0, 2),random.randint(0, 1)
			conv_kernel=NAS_CONFIG['spl']['conv_space']['kernel_size'][ra_c]
			conv_act=NAS_CONFIG['spl']['conv_space']['activation'][ra_d]
			celllist.append(Cell('conv',conv_fiter,conv_kernel,conv_act))
		else:
			ra_a,ra_b=random.randint(0, 1),random.randint(0, 4)
			pool_type=NAS_CONFIG['spl']['pool_space']['pooling_type'][ra_a]
			pool_kernel=NAS_CONFIG['spl']['pool_space']['kernel_size'][ra_b]
			celllist.append(Cell('pooling',pool_type,pool_kernel))
		number-=1
	return celllist
def save_glolog(fp,Net,num):
	s=""
	for x in Net.item_list[num].cell_list:
		x=str(x)
		st=str(x.split(','))
		for x in st:
			s=s+x
	for x in Net.item_list[num].graph:
		x=str(x)
		st=str(x.split(','))
		for x in st:
			s=s+x
	s=s+'\n'
	fp.write(str(Net.item_list[num].score)+' '+s);
	fp.flush()
	return 
def run_global(eva,item,ngpu,gpu_list):
	#print(str(ngpu))
	import base 
	print(item.cell_list,type(item.cell_list[0]))
	score=random.random()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu);
	score=eva.evaluate(item,[])
	#score=random.random();
	#print(str(ngpu)+"  achieve",cell,graph)
	gpu_list.put(ngpu)
	return score
def initialize_ops(NETWORK_POOL,NetworkItem):
	from .predictor import Predictor
	pred=Predictor()
	for network in NETWORK_POOL:  # initialize the full network by adding the skipping and ops to graph_part
		cell, graph ,table= network.spl.sample()
		network.graph_full = graph
		blocks = []
		# for block in network.pre_block:  # get the graph_full adjacency list in the previous blocks
		# 	blocks.append(block[0])  # only get the graph_full in the pre_bock
		pred_ops = pred.predictor(blocks, graph, table)
		table = network.spl.ops2table(pred_ops,table)
		cell ,graph, code=network.spl.sample()
		NetworkItem.graph ,NetworkItem.cell, NetworkItem.code =graph , cell , code 
	return NETWORK_POOL ,NetworkItem
def dis_global(enu,eva,num_gpu):
	pool= Pool(processes=num_gpu);eva.add_data(300);
	manager = multiprocessing.Manager()
	gpu_list = manager.Queue()
	os.remove('1.txt');f1=open("1.txt",'a')
	NETWORK_POOL =enu.enumerate()
	block_num=0;
	for network in NETWORK_POOL:
		network.spl=Sampler(network.graph_template,0)
	#with Pool(1) as p:
		#NETWORK_POOL,Net_item=p.apply(initialize_ops,(NETWORK_POOL,Net_item,));
	Net=NETWORK_POOL[0]
	for gpu in range(0,num_gpu):     #not test first_samper
		gpu_list.put(gpu)
		cell,graph,code =Net.spl.sample();graph.append([]);cell=rand_gen_cell(cell,1)
		Net_item=NetworkItem(gpu,graph,cell,code)
		#print(cell,Net_item.cell_list,type(Net_item.cell_list),type(Net_item.cell_list[0]))
		Net.item_list.append(Net_item)
	print(Net.item_list[0].cell_list,Net.item_list[1].cell_list)
	results=[];i=0
	while i<200:
		while not gpu_list.empty():
			ngpu=gpu_list.get();
			eva_result=pool.apply_async(run_global,args=(eva,Net.item_list[i],ngpu,gpu_list,))
			#print(eva_result,i,ngpu,Net.item_list[i].graph,Net.item_list[i].cell,flush=True);
			results.append(eva_result);i=i+1;
		k=0
		for result in results[i-2:]:
			Net.item_list[i-2+k].score=result.get();
			print(i-2+k,Net.item_list[i-2+k].score,Net.item_list[i-2+k].cell_list,Net.item_list[i-2+k].graph,flush=True);
			Net.spl.update_opt_model(Net.item_list[i-2+k].code,-Net.item_list[i-2+k].score)
			cell,graph,code =Net.spl.sample();graph.append([]);cell=rand_gen_cell(cell,1);Net_item=NetworkItem(i+k,graph,cell,code);
			Net.item_list.append(Net_item)
			save_glolog(f1,Net,i-2+k)
			k=k+1;
		print(len(Net.item_list))
	pool.close()#
	pool.join()#

class Nas:
    def __init__(self):
        #print(ifs.init_ing)
        self.enu = Enumerater()
        self.eva = Evaluator()
        #self.com = Communication()
        #self.pool = pool
if __name__ == '__main__':
	#pool = Pool(processes=NAS_CONFIG['nas_main']["num_gpu"])
	nas = Nas()
	dis_global(nas.enu,nas.eva,2)