import os, pickle, time, random
import numpy as np
import torch, cv2
import torchvision.transforms as T
import data.misc as misc
import data.data_loader as loader

class Cifar10:
    def __init__(self, args, path, load_batch=(1, 2, 3, 4, 5), verbose=True):
        """
        Download at:
            http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        :param path: where cifar-10 dataset located
        :param load_batch: which data_batch you want to load
        """
        
        self.args = args
        self.path = os.path.expanduser(path)
        if not os.path.exists(self.path):
            raise FileExistsError("Please Download cifar-10 at: "
                                  "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                                  "It will start downloading automatically :) ")
        self.load_batch = load_batch
        self.verbose = verbose
        # Images in cifar-10 dataset has shape of (32, 32, 3)
        self.shape = (32, 32, 3)
        # In cifar-10 dataset images can be classified into 10 labels
        self.labels = torch.eye(10)
        # Mean and Std of cifar-10
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        
    def prepare(self):
        data = []
        for _ in self.load_batch:
            start = time.time()
            name = "data_batch_" + str(_)
            with open(os.path.join(self.path, name), "rb") as db:
                dict = pickle.load(db, encoding="bytes")
                dim = max([len(dict[_]) for _ in dict.keys()])
                for i in range(dim):
                    if self.verbose and i%500 is 0:
                        print("{:5d} sample has been loaded".format(i))
                    data.append([dict[misc.str2bytes("data")][i,:],
                                dict[misc.str2bytes("labels")][i]])
            print("--- cost {:.5} second to load DB {:2d}".format(time.time()-start, _))
        self.dataset = data
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = reshape(image)
        image = loader.transform(self.args).augment_image(image)
        image = T.ToTensor()(image)
        image = T.Normalize(self.mean, self.std)(image)
        return image, self.labels[label]
    
def reshape(img):
    img_R = img[0:1024].reshape((32, 32))
    img_G = img[1024:2048].reshape((32, 32))
    img_B = img[2048:3072].reshape((32, 32))
    return np.dstack((img_R, img_G, img_B))
   
# Function for testing
def examine_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.expanduser("~/Pictures/tmp.jpg"), img)
    
    
if __name__ is "__main__":
    from options.base_options import BaseOptions
    args = BaseOptions().initialize()
    CF10 = Cifar10(args, "~/Downloads/cifar-10")
    CF10.prepare()
    batch_list = random.sample(range(len(CF10.dataset)), 32)
    images = []
    labels = []
    for i in batch_list:
        images.append(CF10[i][0])
        labels.append(CF10[i][1])
    img_batch = torch.stack(images)
    label_batch = torch.stack(labels)
    pass