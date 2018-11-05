import os, argparse
from options.imgaug_options import ImgAug


class BaseOptions(ImgAug):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def initialize(self):
        ImgAug.initialize(self)
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="if None, then use CPU mode.")
        
        self.parser.add_argument("--epoch_num", type=int, default=5000,
                                 help="Total training epoch")
        
        # -------------------------------Optimizer------------------------------------
        self.parser.add_argument("--learning_rate", type=float, default=0.001)
        
        # ------------------------------Input Queue-----------------------------------
        self.parser.add_argument("--batch_size", type=int, default=8)
        
        # -------------------------------Input Images------------------------------------
        self.parser.add_argument("--path", type=str, help="the path of dataset")
        self.parser.add_argument("--model", type=str, help="the path where you save the model")
        self.parser.add_argument("--log", type=str, help="the path want to save your log")
        self.parser.add_argument("--extensions", type=list,
                                 default = ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"])

        self.parser.add_argument("--random_order_load", type=bool, default=False,
                                 help="perform randomly select input source from dataset")
        
        # ------------------------------Miscellaneous----------------------------------
        self.parser.add_argument("--log_dir", type=str, help="log location")
        args = self.parser.parse_args()
        return args


if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)