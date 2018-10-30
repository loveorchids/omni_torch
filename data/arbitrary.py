import os, glob, random
import data.misc as misc
import data.mode as mode
import data.data_loader as loader
from torch.utils import data

class Arbitrary(data.Dataset):
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
        assert max([len(sources), len(modes), len(dig_level)]) is\
            min([len(sources), len(modes), len(dig_level)]), \
            "Length of 'sources', 'modes', 'dig_level' must be same."
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
        if "sizes" in options:
            assert type(options["sizes"]) is list
            assert len(options["sizes"]) is len(load_funcs)
            self.sizes = options["sizes"]
        else:
            self.sizes = None
        
    def prepare(self):
        self.dataset = self.load_dataset()
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        #assert type(index) is int, "Arbitrary Dataset index should be int."
        return self.load_item(self.dataset[index])
    
    def load_item(self, item):
        assert len(item) is len(self.load_funcs), "length of item and mode should be same."
        result = []
        # seed is used to keep same image augmentation manner when load things from one item
        seed = random.randint(0, 100000)
        for i in range(len(item)):
            if self.load_funcs[i] is "image":
                if self.sizes:
                    size = self.sizes[i]
                else:
                    size = self.args.img_size
                result.append(loader.read_image(self.args, item[i], seed, size))
            elif callable(self.load_funcs[i]):
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
        data = []
        path = os.path.expanduser(self.args.path)
        assert len(self.sources) is len(self.modes), "sources and modes should be same dimensions."
        input_types = len(self.modes)
        for i in range(input_types):
            sub_paths = []
            for source in self.sources:
                if type(source) is str:
                    sub_paths.append(os.path.join(path, source))
                elif type(source) is tuple or type(source) is list:
                    # In this case, there are multiple inputs in one source,
                    # but we want it to be considered as one file
                    sub_paths.append([os.path.join(path, _) for _ in source])
                else:
                    raise TypeError
            if self.modes[i] is "path":
                data += mode.load_path_from_folder(self.args, len(data), sub_paths[i],
                                                       self.dig_level[i])
            # We can add other modes if we want
            elif callable(self.modes[i]):
                # mode[i] is a function
                data += self.modes[i](self.args, len(data), sub_paths[i], self.dig_level[i])
            else:
                raise NotImplementedError
        dataset = []
        data_pieces = max([len(_) for _ in data])
        for key in range(len(data)):
            data[key] = misc.compliment_dim(data[key], data_pieces)
        for i in range(data_pieces):
            dataset.append([_[i] for _ in data])
        return dataset
