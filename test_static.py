import unittest
from info_str import NAS_CONFIG


class Test_static(unittest.TestCase):
    def judge_int(self, _v):
        self.assertEqual(int, type(_v))

    def judge_str(self, _v):
        self.assertEqual(str, type(_v))

    def judge_float(self, _v):
        self.assertEqual(float, type(_v))

    def judge_list(self, _v):
        self.assertEqual(list, type(_v))

    def test_int_external(self):
        self.judge_int(NAS_CONFIG['subp_debug'])
        self.judge_int(NAS_CONFIG['eva_debug'])
        self.judge_int(NAS_CONFIG['enum_debug'])
        self.judge_int(NAS_CONFIG['ops_debug'])
        self.judge_int(NAS_CONFIG['task_index'])
        self.judge_int(NAS_CONFIG['m_best'])
        self.judge_int(NAS_CONFIG['opt_best_k'])
        self.judge_int(NAS_CONFIG['randseed'])
        self.judge_int(NAS_CONFIG['depth'])
        self.judge_int(NAS_CONFIG['width'])
        self.judge_int(NAS_CONFIG['max_depth'])
        self.judge_int(NAS_CONFIG['num_gpu'])
        self.judge_int(NAS_CONFIG['block_num'])
        self.judge_int(NAS_CONFIG['finetune_threshold'])

    def test_str_external(self):
        self.judge_str(NAS_CONFIG['ps_host'])
        self.judge_str(NAS_CONFIG['worker_host'])
        self.judge_str(NAS_CONFIG['job_name'])
        self.judge_str(NAS_CONFIG['pattern'])

    def test_int_eva(self):
        self.judge_int(NAS_CONFIG['eva']['IMAGE_SIZE'])
        self.judge_int(NAS_CONFIG['eva']['NUM_CLASSES'])
        self.judge_int(NAS_CONFIG['eva']['NUM_EXAMPLES_FOR_TRAIN'])
        self.judge_int(NAS_CONFIG['eva']['NUM_EXAMPLES_PER_EPOCH_FOR_EVAL'])
        self.judge_int(NAS_CONFIG['eva']['batch_size'])
        self.judge_int(NAS_CONFIG['eva']['epoch'])

    def test_float_eva(self):
        self.judge_float(NAS_CONFIG['eva']['INITIAL_LEARNING_RATE'])
        self.judge_float(NAS_CONFIG['eva']['NUM_EPOCHS_PER_DECAY'])
        self.judge_float(NAS_CONFIG['eva']['LEARNING_RATE_DECAY_FACTOR'])
        self.judge_float(NAS_CONFIG['eva']['MOVING_AVERAGE_DECAY'])
        self.judge_float(NAS_CONFIG['eva']['weight_decay'])
        self.judge_float(NAS_CONFIG['eva']['momentum_rate'])

    def test_str_eva(self):
        self.judge_str(NAS_CONFIG['eva']['model_path'])
        self.judge_str(NAS_CONFIG['eva']['learning_rate_type'])

    def test_list_eva(self):
        self.judge_list(NAS_CONFIG['eva']['boundaries'])
        self.judge_list(NAS_CONFIG['eva']['learing_rate'])

    def test_int_spl(self):
        self.judge_int(NAS_CONFIG['spl']['opt_para']['sample_size'])
        self.judge_int(NAS_CONFIG['spl']['opt_para']['budget'])
        self.judge_int(NAS_CONFIG['spl']['opt_para']['positive_num'])
        self.judge_int(NAS_CONFIG['spl']['opt_para']['uncertain_bit'])

    def test_float_spl(self):
        self.judge_float(NAS_CONFIG['spl']['opt_para']['rand_probability'])

    def test_str_spl(self):
        self.judge_str(NAS_CONFIG['spl']['pattern'])
        self.judge_str(NAS_CONFIG['spl']['opt_para']['sample_size#'])
        self.judge_str(NAS_CONFIG['spl']['opt_para']['budget#'])
        self.judge_str(NAS_CONFIG['spl']['opt_para']['positive_num#'])
        self.judge_str(NAS_CONFIG['spl']['opt_para']['rand_probability#'])
        self.judge_str(NAS_CONFIG['spl']['opt_para']['uncertain_bit#'])

    def test_int_space_graph(self):
        self.judge_int(NAS_CONFIG['space']['block_num'])
        self.judge_int(NAS_CONFIG['space']['graph']['depth'])
        self.judge_int(NAS_CONFIG['space']['graph']['width'])
        self.judge_int(NAS_CONFIG['space']['graph']['max_branch_depth'])
        self.judge_int(NAS_CONFIG['space']['skipping_max_dist'])
        self.judge_int(NAS_CONFIG['space']['skipping_max_num'])

    def test_space_ops_conv(self):
        self.judge_list(NAS_CONFIG['space']['ops']['conv']['filter_size'])
        for i in NAS_CONFIG['space']['ops']['conv']['filter_size']:
            self.judge_int(i)

        self.judge_list(NAS_CONFIG['space']['ops']['conv']['kernel_size'])
        for i in NAS_CONFIG['space']['ops']['conv']['kernel_size']:
            self.judge_int(i)

        self.judge_list(NAS_CONFIG['space']['ops']['conv']['activation'])
        for i in NAS_CONFIG['space']['ops']['conv']['activation']:
            self.judge_str(i)

    def test_space_ops_pooling(self):
        self.judge_list(NAS_CONFIG['space']['ops']['pooling']['pooling_type'])
        for i in NAS_CONFIG['space']['ops']['pooling']['pooling_type']:
            self.judge_str(i)

        self.judge_list(NAS_CONFIG['space']['ops']['pooling']['kernel_size'])
        for i in NAS_CONFIG['space']['ops']['pooling']['kernel_size']:
            self.judge_int(i)


if __name__ == "__main__":
    unittest.main()
