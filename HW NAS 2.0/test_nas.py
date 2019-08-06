import pytest
import random
import tensorflow as tf

from nas import Nas

TEST_ELI = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_M_BEST = [(1,2,-1)]
TEST_RAND_NAS = [[1, 7), (1, 5), (1, 7)]
# TEST_RAND_NAS = [[(1, 2), (1, 1), (1, 2)]]

class TestNas():
    @pytest.fixture(params=TEST_M_BEST)
    def setup_nas(self, request):
        m_best, opt_k, seed = request.param
        return Nas(m_best=m_best, opt_best_k=opt_k, randseed=seed)
    
    @pytest.fixture(params=TEST_RAND_NAS)
    def setup_rand_nas(self, request):
        dep_low, dep_up = request.param[0]
        wid_low, wid_up = request.param[1]
        maxbd_low, maxbd_up = request.param[2]
        
        depth = random.randint(dep_low, dep_up)
        width = random.randint(wid_low, wid_up)
        max_bdepth = random.randint(maxbd_low, maxbd_up)
        return Nas(depth=depth, width=width, max_branch_depth=max_bdepth)

    @pytest.mark.parametrize('pool_size_exp', TEST_ELI)
    def test_eliminate(self, setup_nas, pool_size_exp):
        nas = setup_nas
        pool_size = 10 ** pool_size_exp
        pool = [i for i in range(pool_size)]
        scores = [random.random() for i in range(pool_size)]
        scores_bak = scores.copy()
        mid_val = nas._Nas__eliminate(pool, scores)
        
        for i in range(len(pool)):
            num = pool[i]
            assert(scores[i] is scores_bak[num])
            assert(scores[i] >= mid_val)
    
    @pytest.mark.run
    def test_run(self, setup_rand_nas):
        nas = setup_rand_nas
        nas.run()

if __name__ == '__main__':
    dep_low, dep_up = TEST_RAND_NAS[0][0]
    wid_low, wid_up = TEST_RAND_NAS[0][1]
    maxbd_low, maxbd_up = TEST_RAND_NAS[0][2]
    
    depth = random.randint(dep_low, dep_up)
    width = random.randint(wid_low, wid_up)
    max_bdepth = random.randint(maxbd_low, maxbd_up)
    print("d: {0}, w: {1}, mbd: {2}".format(depth, width, max_bdepth))
    # nas = Nas(depth=depth, width=width, max_branch_depth=max_bdepth, randseed=100)
    nas = Nas()
    nas.run()
