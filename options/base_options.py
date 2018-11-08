import os, argparse
from options.imgaug_options import ImgAug


class BaseOptions(ImgAug):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def initialize(self):
        ImgAug.initialize(self)
        self.parser.add_argument("--code_name", type=str, required=True,
                                 help="Anyway, you have to name your experiment.")
        self.parser.add_argument("--cover_exist", action="store_true")
        
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="if None, then use CPU mode.")
        self.parser.add_argument("--epoch_num", type=int, default=2000,
                                 help="Total training epoch")
        self.parser.add_argument("--deterministic_train", type=bool, default=False,
                                 help="Make the training reproducable")
        self.parser.add_argument("--torch_seed", type=int, default=88)
        self.parser.add_argument("--finetune", type=bool, default=False)
        
        # -------------------------------Optimizer------------------------------------
        self.parser.add_argument("--learning_rate", type=float, default=0.001)
        self.parser.add_argument("--loss_weight_momentum", type=float, default=0.9)
        
        # ------------------------------Input Queue--------threa---------------------------
        self.parser.add_argument("--loading_threads", type=int, default=2,
                                 help="threads used to load data to cpu-memory")
        self.parser.add_argument("--batch_size", type=int, default=8)
        
        # -------------------------------Input Images------------------------------------
        self.parser.add_argument("--path", type=str, help="the path of dataset")
        self.parser.add_argument("--model", type=str, help="the path where you save the model")
        self.parser.add_argument("--log", type=str, help="the path want to save your log")
        self.parser.add_argument("--extensions", type=list,
                                 default = ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"])


        self.parser.add_argument("--random_order_load", type=bool, default=False,
                                 help="perform randomly select input source from dataset")
        
        # ------------------------------ Logging ----------------------------------
        self.parser.add_argument("--log_dir", type=str, help="place where you save your training logs")
        self.parser.add_argument("--model_dir", type=str, help="place where you save your training models")
        self.parser.add_argument("--loss_log", type=str, help="place where you save the change of your training loss")
        self.parser.add_argument("--grad_log", type=str, help="place where you save your trend of training gradients")
        args = self.parser.parse_args()
        return args


if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)