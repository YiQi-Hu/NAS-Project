# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:03:47 2019

@author: Chlori
"""
import random
from .sampling.load_configuration import load_conf
from .optimizer import Dimension
import pickle

'''
p：优化得到概率向量,
q:p中概率所属参数类型的个数，filter_count = k,convkernel_count = l,activation_count = m,
type_count = n,poolkernel_count = o
ls:p中概率所属参数类型
'''

class Sampler():
    def __init__(self):
        setting, pros = load_conf()
        self.p = []
        self.parameters_count = self.get_parameters_count(setting)
        self.ls = self.create_ls()
        
        self.pros = Dimension()
        self.pros.set_dimension_size(len(pros))

        self.pros.set_regions(pros, [0 for _ in range(len(pros))])
        pass
        
        
    #更新p
    def renewp(self,newp):
        self.p = newp

    def get_parameters_count(self, setting):
        filter_count, convkernel_count, activation_count = len(setting['conv']['filter_size']['val']), len(setting['conv']['kernel_size']['val']), len(
            setting['conv']['activation']['val'])
        poolingtype_count, poolkernel_count = len(setting['pooling']['pooling_type']['val']), len(setting['pooling']['kernel_size']['val'])
        return filter_count-1, convkernel_count-1, activation_count-1, poolingtype_count-1, poolkernel_count-1
    
    #创建一个二维数组，为p中概率所属的参数类型分类,根据parameters_count生成，不用遍历
    def create_ls(self):
        ls = []
        ls.append([0])
        j = 1
        for i in range(len(self.parameters_count)):
            temp_ls = []
            while (j == self.parameters_count[j]):
                temp_ls.append(j)
                j += 1
            ls.append(temp_ls)
               
        return ls
    
    #第二步采样
    def _sample_step_2(self,probability,r):
        for i in range(len(probability) - 1):
            probability[i + 1] += probability[i]
        ran = random.random()
        for j in range(len(probability)):
            if ran > probability[len(probability) - 1 - j]:
                r.append(len(probability) - j)
                break
        if ran <= probability[0]:
            r.append(0)
        return r

    #采样，conv or pooling
    def _sample_step_1(self):
        r = []
        ran = random.random()
        if ran < self.p[0]:
            r.append(0)
            r = self._sample_step_2(self.p[1:self.parameters_count[0] + 1],r)
            r = self._sample_step_2(self.p[self.parameters_count[0] + 1:sum(self.parameters_count[0:2]) + 1],r)
            r = self._sample_step_2(self.p[sum(self.parameters_count[0:2]) + 1:sum(self.parameters_count[0:3]) + 1],r)
        else:
            r.append(1)
            r = self._sample_step_2(self.p[sum(self.parameters_count[0:3])+ 1:sum(self.parameters_count[0:4])+ 1],r)
            r = self._sample_step_2(self.p[sum(self.parameters_count[0:4]) + 1:sum(self.parameters_count) + 1],r)
        return r

    # 判断概率是否溢出
    def issample(self):
        j = 0
        for i in range(len(self.parameters_count)):
            pros = 0
            while (j == self.parameters_count[j]):
                pros = p[j] + pros
                j += 1
                if pros > 1:
                    return 0
        return 1

    def get_parameters(self,para):
        setting, pros = load_conf()
        if para[0] == 0:
            return 'conv',setting['conv']['filter_size']['val'][para[1]], setting['conv']['kernel_size']['val'][para[2]]\
                ,setting['conv']['activation']['val'][para[3]]
        if para[0] == 1:
            return 'pooling',setting['pooling']['pooling_type']['val'][para[1]], setting['pooling']['kernel_size']['val'][para[2]]

    #采样，最终结果
    def sample(self, n):
        r_ls = []
        if self.issample():#len(self.p) == (sum(self.getnumber())+ 1) and
            for i in range(n):
                r = self._sample_step_1()
                r_ls.append(self.get_parameters(r))
            return r_ls
        else:
            print('error')#概率溢出

    def get_cell_log(self,POOL,PATH,date):
        for i, j in enumerate(POOL):
            s = 'nn_param_'
            s = s + str(i) + '_'
            s = s + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

        
if __name__ == '__main__':

    p = [0.7,0.3,0.3,0.7,0.2,0.4,0.3,0.6,0.2,0.3,0.4,0.3,0.3,0.7,0.2,0.4,0.3,0.6,0.2,0.3,0.4]#优化给出
    newp = [0.8,0.2,0.3,0.6,0.2,0.4,0.3,0.6,0.2,0.3,0.4,0.3,0.3,0.7,0.2,0.4,0.3,0.5,0.2,0.3,0.4]#优化更新
    
    cell_list = []
    #append cell_list
    def renewcl(r):
        if r[0]=='conv':
            u,a, b, c = r
            cell1 = ConvolutionalCell(filter_size=a, kernel_size=b, activation=c)
        elif r[0]=='pooling':
            u,a, b = r
            cell1 = PoolingCell(pool_type=a,pool_size=b)
        cell_list.append(cell1)


    sampling = Sampler()
    #print(sampling.ls,sampling.q)
    
    r = Sampler.sample()#采样返回结果
    print('sample:',r)
    renewcl(r)
    # print('filter_size:',cell_list[0].filter_size)
    
    sampling.renewp(newp)#更新p
    print('new p:',sampling.p)
    newr = sampling.sample(10)# 重新采样
    print('newsample:',newr)
    renewcl(newr)
    # print('new filter_size:',cell_list[1].filter_size)
    
    print(cell_list[0],cell_list[1])
    POOL = pickle.load('log_test')
    sampling.get_cell_log(POOL,"C:/TEST/",'2018')



