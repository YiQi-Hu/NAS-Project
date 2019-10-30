import os


class Logger:
    def __init__(self, path):
        if not os.path.exists("memory"):
            os.mkdir("memory")
        self.path = "memory/" + path

    def save_info(self, net, rd, original_index, network_num):
        with open(self.path, 'a') as f:
            f.write("block_num: {} round: {} network_index: {}/{} number of scheme: {}\n"
                    .format(len(net.pre_block) + 1, rd, original_index + 1, network_num,
                            len(net.score_list)))
            f.write("graph_part:")
            f.write(str(net.graph_part) + "\n")
            for item in zip(net.graph_full_list, net.cell_list, net.score_list):
                f.write("    graph_full:")
                f.write(str(item[0]) + "\n")
                f.write("    cell_list:")
                f.write(str(item[1]) + "\n")
                f.write("    score:")
                f.write(str(item[2]) + "\n")

