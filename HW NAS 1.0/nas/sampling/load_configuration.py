import json
import random
import os

def load_conf():
    f = open(os.getcwd() + "\\nas\\sampling\\configuration")
    setting = json.load(f)
    pros = setting['pros']
    del setting['pros']
    for key in setting.keys():
        obj = setting[key]
        for keys in obj.keys():
            objs = obj[keys]
            pros.extend(objs['pros'])
            del setting[key][keys]['pros']
    return setting,pros


if __name__ == '__main__':
    def getParameters(para):
        setting, pros = load_conf()
        if para[0] == 0:
            return setting['conv']['filter_size']['val'][para[1]], setting['conv']['kernel_size']['val'][para[2]], \
                   setting['conv']['activation']['val'][para[3]]
        if para[0] == 1:
            return setting['pooling']['pooling_type']['val'][para[1]], setting['pooling']['kernel_size']['val'][para[2]]


    def getNum():
        setting, pros = load_conf()
        k, l, m = len(setting['conv']['filter_size']['val']), len(setting['conv']['kernel_size']['val']), len(
            setting['conv']['activation']['val'])
        n, o = len(setting['pooling']['pooling_type']['val']), len(setting['pooling']['kernel_size']['val'])
        return [k - 1, l - 1, m - 1, n - 1, o - 1]

    print(load_conf())
    print(sum(getNum()))