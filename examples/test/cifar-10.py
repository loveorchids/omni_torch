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
import os, time, sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.expanduser("~/Documents/omni_research"))
import omni_torch.utils as omth_util
import omni_torch.examples.test.model as model
import omni_torch.examples.test.presets as presets
import omni_torch.options.options_edict as edict_options
import omni_torch.data.data_loader as omth_loader
import omni_torch.data.path_loader as omth_data_mode
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
import omni_torch.utils.weight_transfer as weight_transfer
from omni_torch.networks.optimizer.adamtf import AdamTF
from omni_torch.options.base_options import BaseOptions
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
from omni_torch.utils.model_summary import  summary
from keras.models import Sequential
from keras.layers import *


def fetch_data(args, source):
    def just_return_it(args, data, seed, size, ops=None):
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
            print(prefix + " --- loss: %8f, accuracy: %8f at epoch %04d/%04d, cost %03f s ---" %
                  (sum([sum(i) for i in epoch_loss]) / len(epoch_loss),
                   sum([sum(i) for i in epoch_measure]) / len(epoch_measure),
                   epoch, args.epoches_per_phase, time.time() - start_time))
        else:
            print(prefix + " --- loss: %8f at epoch %04d/%04d, cost %03f s ---" %
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

def get_keras_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

map_dict = {
    'conv2d_1': ['conv1_1.weight', 'conv1_1.bias'],
    'conv2d_2': ['conv1_2.weight', 'conv1_2.bias'],
    'conv2d_3': ['conv2_1.weight', 'conv2_1.bias'],
    'conv2d_4': ['conv2_2.weight', 'conv2_2.bias'],
    'dense_1': ['fc_layer1.weight', 'fc_layer1.bias'],
    'dense_2': ['fc_layer2.weight', 'fc_layer2.bias'],
}

if __name__ == "__main__":
    args = edict_options.initialize()
    args = omth_util.prepare_args(args, presets.PRESET, options=["general", "unique", "runtime"])
    if args.deterministic_train:
        torch.manual_seed(args.seed)
    device = torch.device("cuda:" + args.gpu_id)

    net = model.CifarNet_Vanilla()
    # net.apply(omth_init.weight_init)
    keras_model = get_keras_model()
    model_path = os.path.join(os.getcwd(), 'models', "cifar10_cnn.h5")
    net = weight_transfer.initialize_with_keras_hdf5(keras_model, map_dict, net, model_path)
    net.to(device)
    summary(net, input_size=(3, 32, 32), device=device)
    
    if args.finetune:
        net = util.load_latest_model(args, net)

    train_set = fetch_data(args, [("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5")])
    test_set = fetch_data(args, ["test_batch"])
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer = AdamTF(net.parameters(), lr=1e-4, weight_decay=1e-6, eps=1e-7)

    for epoch in range(args.epoch_num):
        print("\n ======================== PHASE: %s/%s ========================= " %
              (str(int(args.curr_epoch / args.epoches_per_phase)).zfill(4), str(args.epoch_num).zfill(4)))
        fit(net, args, train_set, device, optimizer, criterion, measure=calculate_accuracy)
        evaluation(net, args, test_set, device, optimizer, criterion, measure=calculate_accuracy)
        omth_util.save_model(args, args.curr_epoch, net.state_dict())
    
