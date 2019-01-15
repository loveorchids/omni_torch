import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as tf
sys.path.append(os.path.expanduser("~/Documents/omni_torch"))
import networks.blocks as block
from data.arbitrary_dataset import Arbitrary_Dataset
import data.data_loader as loader
import data.path_loader as mode


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv_block1_1 = block.Resnet_Block(input=3, filters=[64, 64], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])
        self.conv_block1_2 = block.Resnet_Block(input=64, filters=[64, 64], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])
        

        self.conv_block2_1 = block.Resnet_Block(input=64, filters=[128, 128], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])
        self.conv_block2_2 = block.Resnet_Block(input=128, filters=[128, 128], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])

        self.conv_block3_1 = block.Resnet_Block(input=128, filters=[256, 256], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])
        self.conv_block3_2 = block.Resnet_Block(input=256, filters=[256, 256], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])

        self.conv_block4_1 = block.Resnet_Block(input=256, filters=[512, 512], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])
        self.conv_block4_2 = block.Resnet_Block(input=512, filters=[512, 512], kernel_sizes=[3, 3],
                                                stride=[1, 1], padding=[1, 1], groups=[1, 1])

        self.conv_block5 = block.conv_block(input=512, filters=[512], kernel_sizes=[3], stride=[1],
                                            padding=[1], groups=[1], repeat=1)

        self.fc_layer = block.fc_layer(input=2048, layer_size=[128, 256, 512, 1024, 10], activation=nn.ReLU,
                                       batch_norm=True)

    def forward(self, x):
        x = self.pool(self.conv_block1_2(self.conv_block1_1(x)))
        x = self.pool(self.conv_block2_2(self.conv_block2_1(x)))
        x = self.pool(self.conv_block3_2(self.conv_block3_1(x)))
        x = self.pool(self.conv_block4_2(self.conv_block4_1(x)))
        x = self.conv_block5(x)
        #x = tf.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


def fit(net, args, train_set, val_set, device, optimizer, criterion, finetune=False,
        do_validation=True, validate_every_n_epoch=5):
    high_accu = 0
    accu_list = []
    net.train()
    if finetune:
        model_path = os.path.expanduser(args.model_dir)
        net.load_state_dict(torch.load(model_path))
    for epoch in range(args.epoch_num):
        for batch_idx, (img_batch, label_batch) in enumerate(train_set):
            start_time = time.time()
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            pred_label = torch.max(prediction, 1)[1]
            accu = int(torch.sum(torch.eq(pred_label, label_batch))) / img_batch.size(0) * 100
            accu_list.append(accu)
            loss = criterion(prediction, label_batch)
            print("--- loss: %8f at batch %04d / %04d, cost %03f seconds, accuracy: %03f ---" %
                  (float(loss.data), batch_idx, epoch, time.time() - start_time, accu))
            loss.backward()
            optimizer.step()
        if do_validation:# and epoch % validate_every_n_epoch == 0:
            print("Validating...")
            start_time = time.time()
            val_accu = []
            for batch_idx, (img_batch, label_batch) in enumerate(val_set):
                img_batch, label_batch = img_batch.to(device), label_batch.to(device)
                prediction = net(img_batch)
                pred_label = torch.max(prediction, 1)[1]
                val_accu.append(int(torch.sum(torch.eq(pred_label, label_batch))) /
                                img_batch.size(0) * 100)
            avg_val_accu = sum(val_accu)/len(val_accu)
            if avg_val_accu > high_accu:
                torch.save(net.state_dict(), args.model_dir)
            print("--- cost %03f seconds, validation accuracy: %03f ---" %
                  (time.time() - start_time, avg_val_accu))
    return accu_list

def fetch_data(args, source):
    import multiprocessing as mpi
    def just_return_it(args, data, seed=None, size=None):
        """
        Because the label in cifar dataset is int
        So here it will be transfered to a torch tensor
        """
        return torch.tensor(data)
    print("loading Dataset...")
    data = Arbitrary_Dataset(args=args, load_funcs=[loader.to_tensor, loader.just_return_it],
                             sources=source, modes=[mode.load_cifar_from_pickle], dig_level=[0])
    data.prepare()
    print("loading Completed!")
    kwargs = {'num_workers': mpi.cpu_count(), 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    return data_loader

if __name__ == "__main__":
    print("start0")
    from options.base_options import BaseOptions
    import matplotlib as plt
    print("start")
    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/cifar-10"
    args.model_dir = "~/Documents/cifar10"
    args.batch_size = 256

    device = torch.device("cuda:1")
    cifar = CifarNet()
    cifar.to(device)

    train_set = fetch_data(args, [("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4")])
    #test_set = fetch_data(args, ["test_batch"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cifar.parameters(), lr=args.learning_rate)

    trace = fit(cifar, args, train_set, train_set, device, optimizer, criterion, finetune=False)
    
    fig = plt.figure()
    ax = plt.axes()
    
    ax.plot(trace)
    
    plt.show()
    
    



