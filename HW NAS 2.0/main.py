from nas.nas import Nas

if __name__ == '__main__':
    nas = Nas(m_best=1, opt_best_k=5, randseed=1000, depth=4, width=2, max_branch_depth=3)
    nas.run(pattern="Block",block_num=5)
