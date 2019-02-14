"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import os, time
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
#sys.path.append(os.path.expanduser("~/Documents/omni_research"))
import omni_torch.utils as omth_util
import omni_torch.examples.test.presets as presets
import omni_torch.options.options_edict as edict_options
import omni_torch.networks.blocks as omth_blocks
import omni_torch.networks.initialization as omth_init
import omni_torch.data.data_loader as omth_loader
import omni_torch.data.path_loader as omth_data_mode
import omni_torch.visualize.basic as vb
from omni_torch.networks.optimizer.adamtf import AdamTF
from omni_torch.options.base_options import BaseOptions
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_025 = nn.Dropout2d(0.25)
        self.dropout_050 = nn.Dropout(0.50)
        self.conv_block1 = omth_blocks.conv_block(input=3, filters=[32, 32], kernel_sizes=[3, 3], stride=[1, 1],
                                                  padding=[1, 0], batch_norm=False, dropout=0)
        self.conv_block2 = omth_blocks.conv_block(input=32, filters=[64, 64], kernel_sizes=[3, 3], stride=[1, 1],
                                                  padding=[1, 0], batch_norm=False, dropout=0)
        self.fc_layer1 = omth_blocks.fc_layer(2304, [512], activation=nn.ReLU(), batch_norm=False)
        #self.fc_layer2 = omth_blocks.fc_layer(512, [10], activation=nn.Softmax(), batch_norm=False)
        self.fc_layer2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.conv_block1(x))
        x = self.dropout_025(x)
        x = self.pool(self.conv_block2(x))
        x = self.dropout_025(x)

        x = x.view(x.size(0), -1)
        x = self.dropout_050(self.fc_layer1(x))
        #x = self.softmax(self.fc_layer2(x))
        x = self.fc_layer2(x)
        return x

def fetch_data(args, source):
    def just_return_it(args, data, seed, size, ops=None):
        """
        Because the label in cifar dataset is int
        So here it will be transfered to a torch tensor
        """
        return torch.tensor(data, dtype=torch.long)
    import multiprocessing as mpi
    print("loading Dataset...")
    data = Arbitrary_Dataset(args=args, load_funcs=[omth_loader.to_tensor, just_return_it],
                             sources=source, modes=[omth_data_mode.load_cifar_from_pickle], dig_level=[0])
    data.prepare()
    print("loading Completed!")
    kwargs = {'num_workers': mpi.cpu_count(), 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    return data_loader

def fit(net, args, dataset, device, optimizer, criterion, measure=None, is_train=True,
        visualize_step=100, visualize_op=None, visualize_loss=False,
        plot_low_bound=None, plot_high_bound=None):
    if is_train:
        net.train()
        prefix = ""
        iter = args.epoches_per_phase
    else:
        net.eval()
        prefix = "VAL"
        iter = 1
    Basic_Losses, Basic_Measures = [], []
    if "loss_name" in args.items():
        loss_name = args.loss_name
    else:
        loss_name = ["loss"]
    for epoch in range(iter):
        epoch_loss, epoch_measure = [], []
        start_time = time.time()
        for batch_idx, data in enumerate(dataset):
            if args.steps_per_epoch is not None and batch_idx >= args.steps_per_epoch:
                break
            img_batch, label_batch = data[0].to(device), data[1].to(device)
            prediction = net(img_batch)
            # Tranform prediction and label into list in case there are multiple input and output sources
            prediction = [prediction] if type(prediction) is not list else prediction
            label_batch = [label_batch] if type(label_batch) is not list else label_batch
            basic_loss = [criterion(prediction[i], label) for i, label in enumerate(label_batch)]
            Basic_Losses.append([float(loss.data) for loss in basic_loss])
            epoch_loss.append(Basic_Losses[-1])
            if measure:
                basic_measure = [measure(prediction[i], label) for i, label in enumerate(label_batch)]
                Basic_Measures.append([float(loss) for loss in basic_measure])
                epoch_measure.append(Basic_Measures[-1])
            if is_train:
                optimizer.zero_grad()
                total_loss = sum([loss for loss in basic_loss])
                total_loss.backward()
                optimizer.step()
            # Visualize Output and Loss
            assert len(loss_name) == len(Basic_Losses[-1]), \
                "please check your args.loss_name for it does not match the length of the number of loss produced by the network"
            loss_dict = dict(zip(loss_name, Basic_Losses[-1]))
            if batch_idx % visualize_step == 0 and visualize_op is not None:
                visualize_op(args, batch_idx, prefix, loss_dict, img_batch, prediction, label_batch)
        if measure:
            print(prefix + " --- total loss: %8f, total measure: %8f at epoch %04d/%04d, cost %3f seconds ---" %
                  (sum([sum(i) for i in epoch_loss]) / len(epoch_loss),
                   sum([sum(i) for i in epoch_measure]) / len(epoch_measure),
                   epoch, args.epoches_per_phase, time.time() - start_time))
        else:
            print(prefix + " --- total loss: %8f at epoch %04d/%04d, cost %3f seconds ---" %
                  (sum([sum(i) for i in epoch_loss]) / len(epoch_loss),
                   epoch, args.epoches_per_phase, time.time() - start_time))
        if is_train:
            args.curr_epoch += 1
    all_losses = list(zip(*Basic_Losses))
    all_measures = list(zip(*Basic_Measures))
    #print(*list(zip(loss_name, [sum(loss) / len(loss) for loss in all_losses])))
    if is_train and visualize_loss:
        vb.visualize_gradient(args, net)
        vb.plot_loss_distribution([np.asarray(loss) for loss in all_losses], loss_name, args.loss_log,
                                  prefix + "loss_at_", args.curr_epoch, window=11, fig_size=(5, 5),
                                  low_bound=plot_low_bound, high_bound=plot_high_bound)
    return all_losses, all_measures


def evaluation(net, args, val_set, device, optimizer, criterion, measure=None, is_train=False,
               visualize_step=100, visualize_op=None, visualize_loss=False,
               plot_low_bound=None, plot_high_bound=None):
    with torch.no_grad():
        return fit(net, args, val_set, device, optimizer, criterion, measure, is_train,
                   visualize_step, visualize_op, visualize_loss, plot_low_bound, plot_high_bound)

def calculate_accuracy(prediction, label_batch):
    _, pred_idx = torch.max(prediction, dim=1)
    accuracy = sum([1 if int(pred_idx[i]) == int(label_batch[i]) else 0 for i in range(pred_idx.size(0))])
    return accuracy / label_batch.size(0)


if __name__ == "__main__":
    settings = BaseOptions().initialize()
    args = edict_options.initialize()
    if settings.which is None:
        settings.which = ["general", "unique", "runtime"]
    args = omth_util.prepare_args(args, presets.PRESET, options=settings.which)
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:" + args.gpu_id)

    net = CifarNet()
    net.to(device)
    summary(net, input_size=(3, 32, 32))
    net.apply(omth_init.weight_init)

    train_set = fetch_data(args, [("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5")])
    test_set = fetch_data(args, ["test_batch"])
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamTF(net.parameters(), lr=1e-4, weight_decay=1e-6)
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0, eps=1e-7)
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epoch_num):
        print("\n ======================== PHASE: %s/%s ========================= " %
              (str(int(args.curr_epoch / args.epoches_per_phase)).zfill(4), str(args.epoch_num).zfill(4)))
        fit(net, args, train_set, device, optimizer, criterion, measure=calculate_accuracy)
        evaluation(net, args, test_set, device, optimizer, criterion, measure=calculate_accuracy)
        omth_util.save_model(args, args.curr_epoch, net.state_dict())
    
