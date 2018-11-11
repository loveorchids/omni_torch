import argparse
from options.imgaug_options import ImgAug


class BaseOptions(ImgAug):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def initialize(self):
        ImgAug.initialize(self)
        self.parser.add_argument("--code_name", type=str, required=True,
                                 help="Anyway, you have to name your experiment.")
        self.parser.add_argument("--cover_exist", action="store_true")
        
        # -------------------------------Training------------------------------------
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="which gpu you want to use, multi-gpu is not supported here")
        self.parser.add_argument("--epoch_num", type=int, default=2000,
                                 help="Total training epoch")
        self.parser.add_argument("--deterministic_train", type=bool, default=False,
                                 help="Make the training reproducable")
        self.parser.add_argument("--seed", type=int, default=88,
                                 help="If Deterministic train is allowed, this will be the random seed")
        
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--learning_rate", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-4)
        self.parser.add_argument("--batch_norm", type=bool, default=True)
        self.parser.add_argument("--finetune", type=bool, default=False)
        
        # Models does not need to be specified unless you are going to use them
        self.parser.add_argument("--model1", type=str, help="Model you want to choose")
        self.parser.add_argument("--model2", type=str, help="Model you want to choose")
        self.parser.add_argument("--model3", type=str, help="Model you want to choose")
        self.parser.add_argument("--model4", type=str, help="Model you want to choose")
        self.parser.add_argument("--model5", type=str, help="Model you want to choose")
        self.parser.add_argument("--model6", type=str, help="Model you want to choose")

        self.parser.add_argument("--general_options", type=str, help='general settings for a '
                                                            'model that specifies the settings in the package [options]. '
                                                            'It can be both number or path')
        self.parser.add_argument("--unique_options", type=str, help="unique settings for a "
                                                            "particular model that are useless to others. "
                                                            "It can be both number or path")
        
        # ------------------------------MISC-----------------------------------
        self.parser.add_argument("--loading_threads", type=int, default=2,
                                 help="threads used to load data to cpu-memory")
        self.parser.add_argument("--random_order_load", type=bool, default=False,
                                 help="ingore the correspondence of input and output data when load the dataset")
        self.parser.add_argument("--path", type=str, help="the path of dataset")
        self.parser.add_argument("--extensions", type=list,
                                 default = ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"])
        
        args = self.parser.parse_args()
        return args


if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)