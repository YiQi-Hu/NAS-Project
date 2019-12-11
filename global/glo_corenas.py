import copy
import multiprocessing
from multiprocessing import Pool
from numpy import zeros
import nas
import random
import os,copy
#from nas import _gpu_eva
from sampler import Sampler
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
def run_global(eva,item,ngpu,gpu_list):
	#print(str(ngpu))
	score=random.random()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu);
	#score=eva.evaluate(item,[])
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
	manager = multiprocessing.Manager()
	gpu_list = manager.Queue()
	os.remove('1.txt');f1=open("1.txt",'a')
	NETWORK_POOL =enu.enumerate()
	block_num=0
	for network in NETWORK_POOL:
		network.spl=Sampler(network.graph_template,0)
	Net_item=NetworkItem(0)
	#with Pool(1) as p:
		#NETWORK_POOL,Net_item=p.apply(initialize_ops,(NETWORK_POOL,Net_item,));
	Net=NETWORK_POOL[0]
	for gpu in range(0,num_gpu):     #not test first_samper
		gpu_list.put(gpu)
		Net_item.cell ,Net_item.graph, Net_item.code =Net.spl.sample();tmp=copy.deepcopy(Net_item)
		Net.item_list.append(tmp)
	pool= Pool(processes=num_gpu);eva.add_data(300);
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
			print(i-2+k,Net.item_list[i-2+k].score,Net.item_list[i-2+k].cell,Net.item_list[i-2+k].graph,flush=True);
			Net.spl.update_opt_model(Net.item_list[i-2+k].code,-Net.item_list[i-2+k].score)
			Net_item.cell ,Net_item.graph, Net_item.code =Net.spl.sample();tmp=copy.deepcopy(Net_item)
			Net.item_list.append(tmp)
			save_glolog(f1,Net,i-2+k)
			k=k+1;
	pool.close()#
	pool.join()#
