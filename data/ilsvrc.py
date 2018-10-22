import os
import data.misc as misc

class ILSVRC:
    def __init__(self, root, extensions=None, verbose=False):
        self.root = root
        if extensions:
            self.extensions = extensions
        else:
            self.extensions = ("jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP")
        self.verbose = verbose
        self.dataset = self.load_dataset()

    def __getitem__(self, item):
        assert type(item) is int, "Arbitrary Dataset index should be int."
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
    
    def load_dataset(self):
        """
        :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
        """
        dataset = {}
        path = os.path.expanduser(self.root)
        classes = os.listdir(path)
        classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
        for i, cls in enumerate(classes):
            if self.verbose:
                print('Loading {}th {} class.'.format(i, cls))
            dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                                  misc.extension_check(_, self.extensions)]})
        print("Dataset loading is complete.")
        return dataset