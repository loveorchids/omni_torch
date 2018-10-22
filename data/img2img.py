import os
import data.misc as misc

class Img2Img:
    def __init__(self, root, A_B_folder=None, one_to_one=True,
                 extensions=None, verbose=False):
        """
        :param root: path of the dataset root
        :param A_B_folder: name of input and output folder
        :param one_to_one: if the A & B folder is one-to-one correspondence
        :param extensions: extentions that you treat them as the images.
        :param verbose: display the detail of the process
        """
        self.root = root
        if A_B_folder:
            self.A_B_folder = A_B_folder
        else:
            self.A_B_folder = ("trainA", "trainB")
        self.one_to_one = one_to_one
        if extensions:
            self.extensions = extensions
        else:
            self.extensions = ("jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP")
        self.verbose = verbose
        self.dataset = self.img2img_dataset()

    def __getitem__(self, item):
        assert type(item) is int, "Img2Img Dataset index should be int."
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def load_item(self, item, mode):
        assert len(item) is len(mode), "length of item and mode should be same."
        result = []
        for i in range(len(item)):
            if callable(mode):
                result.append(mode[i](item[i]))
            else:
                raise NotImplementedError
        return result

    def img2img_dataset(self):
        dataset = []
        path = os.path.expanduser(self.root)
        assert len(self.A_B_folder) == 2, "A_B_folder should be the name of source and target folder."
        source = os.path.join(path, self.A_B_folder[0])
        target = os.path.join(path, self.A_B_folder[1])
        assert os.path.isdir(source) and os.path.isdir(target), "one of the folder does not exist."
        source_imgs = [os.path.join(path, self.A_B_folder[0], _) for _ in os.listdir(source)
                       if misc.extension_check(_, self.extensions)]
        target_imgs = [os.path.join(path, self.A_B_folder[1], _) for _ in os.listdir(target)
                       if misc.extension_check(_, self.extensions)]
        if self.one_to_one:
            assert len(source_imgs) == len(target_imgs)
            if self.verbose: print("Sorting files...")
            source_imgs.sort()
            target_imgs.sort()
        else:
            dim = max(len(source_imgs), len(target_imgs))
            if len(source_imgs) < len(target_imgs):
                source_imgs = misc.compliment_dim(source_imgs, dim)
            else:
                target_imgs = misc.compliment_dim(target_imgs, dim)
        for i in range(len(source_imgs)):
            if self.verbose and i % 100 == 0:
                print("{} samples has been loaded...".format(i))
            dataset.append([source_imgs[i], target_imgs[i]])
        print('Dataset loading is complete.')
        return dataset