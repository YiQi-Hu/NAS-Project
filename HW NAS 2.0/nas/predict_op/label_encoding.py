pool_type = {'pool max': 0, 'pool avg': 1}
pool_size = {'2': 0, '3': 1, '4': 2, '5': 3, '7': 4,'global':5}
conv_size = {'1': 0, '3': 1, '5': 2, '7': 3, '11': 4}
conv_filters = {'16': 0, '32': 1, '64': 2, '48': 3, '96': 4, '128': 5, '192': 6, '256': 7, '512': 8, '1024': 9}
conv_act = {'0': 0, 'relu': 1}
conv_bn = {'0': 0, 'bn':1}
conv_scale = {'0': 0, 'scale': 1}

pool_type_rev = {v: k for k, v in pool_type.items()}
pool_size_rev = {v: k for k, v in pool_size.items()}
conv_size_rev = {v: k for k, v in conv_size.items()}
conv_filters_rev = {v: k for k, v in conv_filters.items()}
conv_act_rev = {v: k for k, v in conv_act.items()}
conv_bn_rev = {v: k for k, v in conv_bn.items()}
conv_scale_rev = {v: k for k, v in conv_scale.items()}


# node_op 一个网络中所有节点的操作配置，用一个列表存储
# 列表的每一个元素形如：[int,[]]  [节点类型0,1,2,  [详细参数]]
# 0 pool，1 conv
def encoder(node_op):
    label = []
    for i in range(len(node_op)):
        if node_op[i][0] == 0:
            type = node_op[i][1][0]
            size = node_op[i][1][1]
            label.append(pool_type[type]*len(pool_size)+pool_size[size])
        if node_op[i][0] == 1:
            shift = len(pool_type)*len(pool_size)
            size = node_op[i][1][0]
            filters = node_op[i][1][1]
            act = node_op[i][1][2]
            bn = node_op[i][1][3]
            scale = node_op[i][1][4]
            step1 = len(conv_scale)
            step2 = len(conv_bn)*step1
            step3 = len(conv_act)*step2
            step4 = len(conv_filters)*step3
            label.append(conv_size[size]*step4+conv_filters[filters]*step3+conv_act[act]*step2+conv_bn[bn]*step1+conv_scale[scale]+shift)
    return label


def decoder(label):
    node_op = []
    shift = len(pool_type)*len(pool_size)
    for e in label:
        para = []
        if e < shift:
            para.append('pooling')

        else:
            e = e-shift
            step1 = len(conv_scale)
            step2 = len(conv_bn)*step1
            step3 = len(conv_act)*step2
            step4 = len(conv_filters)*step3
            size = int(e/step4)
            e = e % step4
            filters = int(e/step3)
            e = e % step3
            act = int(e/step2)
            e = e % step2
            bn = int(e/step1)
            scale = e % step1
            para.append(conv_filters_rev[filters])
            para.append(conv_size_rev[size])

        node_op.append(para)
    return node_op


def getClassNum():
    return len(pool_type)*len(pool_size)+len(conv_size)*len(conv_filters)*len(conv_act)*len(conv_bn)*len(conv_scale)


