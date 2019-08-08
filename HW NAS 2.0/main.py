from .nas import Nas

if __name__ == '__main__':
    nas = Nas(randseed=1000)
    nas.run(pattern="Block",block_num=5)