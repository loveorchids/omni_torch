import os, time
import torch, cv2
import torch.nn as nn
from torch.autograd import Variable
import networks.blocks as block
from data.img2img import Img2Img
import data.data_loader as loader
import visualize.basic as vb


class BuddhaNet(nn.Module):
    def __init__(self):
        super(BuddhaNet, self).__init__()

        self.down_conv1 = block.conv_block(1, [48, 128, 128], 1, kernel_sizes=[5, 3, 3],
                                            stride=[2, 1, 1], padding=[2, 1, 1], groups=[1]*3,
                                           activation=nn.SELU)
        self.down_conv2 = block.conv_block(128, [256, 256, 256], 1, kernel_sizes=[3, 3, 3],
                                            stride=[2, 1, 1], padding=[1]*3, groups=[1]*3,
                                           activation=nn.SELU)
        self.down_conv3 = block.conv_block(256, [256, 512, 1024, 1024, 1024, 1024, 512, 256],
                                           1, kernel_sizes=[3] * 8, stride=[2] + [1] * 7,
                                           padding = [1] * 8, groups = [1] * 8,
                                           activation=nn.SELU)
        self.up_conv1 = block.conv_block(256, [256, 256, 128], 1, kernel_sizes=[4, 3, 3],
                                           stride=[0.5, 1, 1], padding=[1] * 3, groups=[1] * 3,
                                           activation=nn.SELU)
        self.up_conv2 = block.conv_block(128, [128, 128, 48], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1]*3, groups=[1]*3,
                                           activation=nn.SELU)
        self.up_conv3 = block.conv_block(48, [48, 24, 1], 1, kernel_sizes=[4, 3, 3],
                                         stride=[0.5, 1, 1], padding=[1]*3, groups=[1]*3,
                                           activation=nn.SELU)

    def forward(self, input):
        out = self.down_conv1(input)
        out = self.down_conv2(out)
        out = self.down_conv3(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        return out
    
    def semantic(self, input):
        out = self.down_conv1(input)
        out = self.down_conv2(out)
        return out


def fit(net, args, data_loader, device, optimizer, criterion, finetune=False,
        do_validation=True, validate_every_n_epoch=5):
    net.train()
    if finetune:
        model_path = os.path.expanduser(args.model)
        net.load_state_dict(torch.load(model_path))
    for epoch in range(args.epoch_num):
        for batch_idx, (img_batch, label_batch) in enumerate(data_loader):
            start_time = time.time()
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            prediction = net(img_batch)
            if batch_idx % 100 == 0:
                vb.vis_image(args, [img_batch, prediction, label_batch],
                             epoch, batch_idx, idx=0)
                #pass
            loss = criterion(prediction, label_batch)
            print("--- loss: %8f at batch %04d / %04d, cost %3f seconds ---" %
                  (float(loss.data), batch_idx, epoch, time.time() - start_time))
            loss.backward()
            optimizer.step()
        #os.system("nvidia-smi")

        if do_validation and epoch % validate_every_n_epoch == 0:
            pass
            # test(net, args, data_loader, device)
        if epoch % 100 == 0:
            model_path = os.path.expanduser(args.model)
            torch.save(net.state_dict(), model_path + "_" + str(epoch))

def calculate_loss(net, prediction, label, device, decay):
    l2_reg = Variable(torch.zeros(1).to(device), requires_grad=True)
    for W in net.parameters():
        l2_reg = l2_reg + W.norm(2)
    mse_loss = nn.MSELoss()
    return mse_loss(prediction, label) + l2_reg * decay

def fetch_data(args, sources):
    import multiprocessing as mpi
    data = Img2Img(args=args, sources=sources, modes=["path"] * 2,
                   load_funcs=[loader.read_image] * 2, dig_level=[0] * 2)
    data.prepare()
    works = mpi.cpu_count() - 2
    kwargs = {'num_workers': works, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return data_loader


def test_fetch_data(args, sources):
    data_loader = fetch_data(args, sources)
    print("Test 5 data batches...")
    for batch_idx, (img_batch, label_batch) in enumerate(data_loader):
        if batch_idx >= 5:
            break
    print("Test completed successfully!")


if __name__ == "__main__":
    from options.base_options import BaseOptions

    args = BaseOptions().initialize()
    args.path = "~/Pictures/dataset/buddha"
    args.model = "~/Pictures/dataset/buddha/models/train_model"
    args.log = "~/Pictures/dataset/buddha/logs"
    args.img_channel = 1
    args.batch_size = 1

    device = torch.device("cuda:1")
    buddhanet = BuddhaNet()
    buddhanet.to(device)

    train_set = fetch_data(args, ["groupa", "groupb"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(buddhanet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    fit(buddhanet, args, train_set, device, optimizer, criterion, finetune=False)



