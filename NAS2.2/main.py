import json
from multiprocessing import Pool
from nas.nas import Nas


def load_search_space(pattern):
    if pattern == "Block":
        path = "./parameters/search_space_block"
    else:
        path = "./parameters/search_space_global"
    with open(path, "r", encoding="utf-8") as f:
        space = json.load(f)
    return space


def load_config(pattern):
    if pattern == "Block":
        path = "./parameters/config_block"
    else:
        path = "./parameters/config_global"
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    PATTERN = "Global"
    setting = load_config(PATTERN)
    search_space = load_search_space(PATTERN)
    nas = Nas(PATTERN, setting, search_space)
    result = nas.run()
    print(result)
