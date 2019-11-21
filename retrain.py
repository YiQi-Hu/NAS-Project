import os

from base import Cell, NetworkItem
from evaluator import Evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
	
	graph_full=[[1, 2, 6], [2, 4, 8], [3, 6, 7, 8], [4, 6, 7], [5, 7, 8], [6, 7, 9], [7, 9, 12],
	[8, 10, 11, 13], [9, 10, 11, 12], [10, 12, 14], [11, 15], [12, 14, 16], [13, 15, 17, 19],
	[14, 16, 17], [15, 16, 19], [16, 17], [17, 19], [18, 19], [19], []]
	cellist=[('conv', 64, 3, 'leakyrelu'), ('conv', 48, 3, 'leakyrelu'), ('conv', 32, 3, 'relu'),
	('pooling', 'max', 7), ('pooling', 'max', 6), ('conv', 128, 3, 'relu'),
	('conv', 48, 3, 'leakyrelu'), ('pooling', 'max', 7), ('conv', 256, 3, 'relu'),
	('conv', 256, 5, 'leakyrelu'), ('pooling', 'max', 6), ('conv', 192, 1, 'relu'),
	('conv', 512, 3, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('pooling', 'max', 8),
	('conv', 48, 1, 'relu'), ('pooling', 'avg', 7), ('conv', 192, 1, 'leakyrelu'), ('pooling', 'avg', 8),
	('pooling', 'max', 6)]
	cell_list=[]
	for x in cellist:
		if len(x)==4:
			cell_list.append(Cell(x[0],x[1],x[2],x[3]))
		else:
			cell_list.append(Cell(x[0],x[1],x[2]))
	print(cell_list)
	network = NetworkItem(0, graph_full, cell_list, "")
	eva=Evaluator()
	eva.add_data(40000)
	eva.evaluate(network,[])
