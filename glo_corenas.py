import copy
import multiprocessing
from multiprocessing import Pool
from numpy import zeros
import nas
import random
import os
#from nas import _gpu_eva
from base import Network ,NetworkItem
def save_glolog(fp,Net,num):
	s=""
	for x in Net.item_list[num].cell:
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
def run_global(eva,graph,cell,ngpu,gpu_list):
	print(str(ngpu))
	score=random.random()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu);
	#score=eva.evaluate(graph,cell,[])
	#score=random.random();
	print(str(ngpu)+"  achieve",cell,graph)
	gpu_list.put(ngpu)
	return score
def initialize_ops(NETWORK_POOL,NetworkItem):
	from .predictor import Predictor
	pred=Predictor()
	for network in NETWORK_POOL:  # initialize the full network by adding the skipping and ops to graph_part
		cell, graph = network.spl.sample()
		network.graph_full = graph
		blocks = []
		# for block in network.pre_block:  # get the graph_full adjacency list in the previous blocks
		# 	blocks.append(block[0])  # only get the graph_full in the pre_bock
		graph , cell , code= pred.first_sample(blocks, graph)  #first_sample to do
		NetworkItem.graph ,NetworkItem.cell, NetworkItem.code =graph , cell , code 
	return NETWORK_POOL ,NetworkItem
def dis_global(enu,eva,num_gpu):
	manager = multiprocessing.Manager()
	gpu_list = manager.Queue()
	f1=open("1.txt",'a')
	NETWORK_POOL =enu.enumerate()
	block_num=0
	for network in NETWORK_POOL:
		network.init_sample(0)
	Net_item=NetworkItem(0)
	#with Pool(1) as p:
		#NETWORK_POOL,Net_item=p.apply(initialize_ops,(NETWORK_POOL,Net_item,));
	nn=NETWORK_POOL[0];
	Net=Network(0,nn.graph_part);
	for gpu in range(0,num_gpu):     #not test first_samper
		gpu_list.put(gpu)
		Net_item.cell ,Net_item.graph, Net_item.code =nn.spl.sample()
		Net.item_list.append(Net_item)
	pool= Pool(processes=num_gpu);eva.add_data(300);
	results=[];i=0
	while i<200:
		while not gpu_list.empty():
			ngpu=gpu_list.get();
			eva_result=pool.apply_async(run_global,args=(eva,Net.item_list[i].graph,Net.item_list[i].cell,ngpu,gpu_list,))
			print(eva_result,i,ngpu,Net.item_list[i].graph,Net.item_list[i].cell,flush=True);
			results.append(eva_result);i=i+1;
		k=0
		for result in results[i-2:]:
			Net.item_list[i-2+k].score=result.get();#print("score",score)
			nn.spl.update_model(Net.item_list[i-2+k].code,Net.item_list[i-2+k].score)
			Net_item.cell ,Net_item.graph, Net_item.code =nn.spl.sample()
			Net.item_list.apennd(Net_item)
			save_glolog(f1,Net,i-2+k)
			k=k+1;
	pool.close()#
	pool.join()#
