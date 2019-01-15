import os, glob, random
import omni_torch.data as data
import omni_torch.data.misc as misc
import omni_torch.data.path_loader as mode
import omni_torch.data.data_loader as loader
from torch.utils import data as tud

ALLOW_WARNING = data.ALLOW_WARNING

class Arbitrary_Dataset(tud.Dataset):
    def __init__(self, args, sources, modes, load_funcs, dig_level, loader_ops=None, **options):
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

        # data_types represent the number of input source and output labels
        data_types = len(load_funcs)

        if loader_ops:
            assert len(loader_ops) is data_types
            self.loader_ops = loader_ops
        else:
            self.loader_ops = [None] * data_types

        if "verbose" in options:
            assert type(options["verbose"]) is bool
            self.verbose = options["verbose"]
        else:
            self.verbose = False
        if "sizes" in options:
            assert type(options["sizes"]) is list
            assert len(options["sizes"]) is data_types
            self.sizes = options["sizes"]
        else:
            self.sizes = [args.final_size] * data_types
        
    def prepare(self):
        self.dataset = self.load_dataset()
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        result = []
        if self.args.random_order_load:
            items = []
            item_num = len(self.dataset[0])
            up_range = len(self.dataset)
            selected_j = []
            for i in range(item_num):
                j = random.randint(0, up_range-1)
                # avoid repeated elements
                while j in selected_j:
                    #print("repeat element ENCOUNTERED!")
                    j = random.randint(0, up_range - 1)
                selected_j.append(j)
                items.append(self.dataset[j][i])
        else:
            items = self.dataset[index]
        assert len(items) is len(self.load_funcs), "length of item and mode should be same."
        if self.args.deterministic_train:
            seed = index + self.args.curr_epoch
        else:
            # seed is used to keep same image augmentation manner when load things from one item
            seed = random.randint(0, 100000)
        for i in range(len(items)):
            if self.load_funcs[i] is "image":
                result.append(loader.read_image(self.args, items[i], seed, self.sizes[i], self.loader_ops[i]))
            if self.load_funcs[i] is "multiimages":
                result += loader.read_image(self.args, items[i], seed, self.sizes[i], self.loader_ops[i])
            elif callable(self.load_funcs[i]):
                result.append(self.load_funcs[i](self.args, items[i], seed, self.sizes[i], self.loader_ops[i]))
            else:
                raise TypeError
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
            if self.modes[i] == "path":
                data += mode.load_path_from_folder(self.args, len(data), sub_paths[i],
                                                       self.dig_level[i])
            elif self.modes[i] == "sep_path":
                data += mode.load_path_from_multi_folder(self.args, len(data), sub_paths[i],
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
