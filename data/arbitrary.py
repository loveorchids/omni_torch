import os, glob, random
import data.misc as misc

class Arbitrary:
    def __init__(self, args, sources, modes, load_funcs, dig_level, **options):
        """
        A generalized data initialization method, inherited by other data class, e.g. ilsvrc, img2img, etc.
        Arbitrary is a parent class of all other data class.

        :param args: options from terminal
        :param sources: a list of input and output source folder or file(e.g. csv, mat, xml, etc.)
        :param modes: a list of how researchers wanted to load infomation from :param sources
        :param load_funcs: infomation loaded from :param source is not guaranteed to be able to feed
        the neural network, thus they need to be loaded as a inputable form. e.g. paths should be loaded
        as image tensors
        :param dig_level: when loading path from a folder, it might contains subfolders, dig_level enables
        you to load as deep as you want to.
        :param options: For Future upgrade.
        """
        assert max([len(sources), len(modes), len(load_funcs), len(dig_level)]) is\
            min([len(sources), len(modes), len(load_funcs), len(dig_level)]), \
            "Length of 'sources', 'modes', 'load_funcs', 'dig_level' must be same."
        self.args = args
        self.sources = sources
        self.modes = modes
        self.load_funcs = load_funcs
        self.dig_level = dig_level
        if "verbose" in options:
            assert type(options["verbose"]) is bool
            self.verbose = options["verbose"]
        else:
            self.verbose = False
        
    def prepare(self):
        self.dataset = self.load_dataset()

    def fetch_batch(self):
        
        
    def __getitem__(self, index):
        #assert type(index) is int, "Arbitrary Dataset index should be int."
        return self.load_item(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)
    
    def load_item(self, item):
        assert len(item) is len(self.load_funcs), "length of item and mode should be same."
        result = []
        seed = random.randint(0, 100000)
        for i in range(len(item)):
            if callable(self.load_funcs[i]):
                result.append(self.load_funcs[i](self.args, item[i], seed))
            else:
                raise NotImplementedError
        return result

    def load_dataset(self):
        """
        :param path: dataset's root folder
        :param sources: all the sub-folders or files you want to read(correspond to data_load_funcs)
        :param data_load_funcs: the way you treat your sub-folders and files(correspond to folder_names)
        :param dig_level: how deep you want to find the sub-folders
        :return: a dataset in the form of a list and each element is an input output pair
        """
        data = {}
        path = os.path.expanduser(self.args.path)
        assert len(self.sources) is len(self.modes), "sources and modes should be same dimensions."
        input_types = len(self.modes)
        for i in range(input_types):
            sub_paths = [os.path.join(path, _) for _ in self.sources]
            if self.modes[i] is "path":
                data.update(self.load_path_from_folder(len(data), sub_paths[i], self.dig_level[i]))
            # We can add other modes if we want
            elif callable(self.modes[i]):
                # mode[i] is a function
                data.update(self.modes[i](len(data), sub_paths[i], self.dig_level[i]))
            else:
                raise NotImplementedError
        dataset = []
        data_pieces = max([len(_) for _ in data.values()])
        for key in data.keys():
            data[key] = misc.compliment_dim(data[key], data_pieces)
        for i in range(data_pieces):
            dataset.append([_[i] for _ in data.values()])
        return dataset
    
    @staticmethod
    def load_path_from_csv(args, len, paths, dig_level=0):
        if type(paths) is str:
            paths = [paths]
        for path in paths:
            with open(path, "r") as csv_file:
                pass
    
    @staticmethod
    def load_path_from_folder(args, len, paths, dig_level=0):
        """
        'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
        'dig_level' represent how deep you want to get paths from.
        """
        output = []
        if type(paths) is str:
            paths = [paths]
        for path in paths:
            current_folders = [path]
            # Do not delete the following line, we need this when dig_level is 0.
            sub_folders = []
            while dig_level > 0:
                sub_folders = []
                for sub_path in current_folders:
                    sub_folders += glob.glob(sub_path + "/*")
                current_folders = sub_folders
                dig_level -= 1
            sub_folders = []
            for _ in current_folders:
                sub_folders += glob.glob(_ + "/*")
            output += sub_folders
        if args.extensions:
            output = [_ for _ in output if misc.extension_check(_, args.extensions)]
        # 1->A, 2->B, 3->C, ..., 26->Z
        key = misc.number_to_char(len)
        return {key: output}

