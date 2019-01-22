import time
import torch
import omni_torch.visualize.basic as vb
import numpy as np

def fit(net, args, dataset, device, optimizer, criterion, measure=None, is_train=True,
        visualize_step=100, visualize_op=None, visualize_loss=False,
        plot_low_bound=None, plot_high_bound=None):
    """
    
    :param net: The network you defined
    :param args:
    :param dataset:
    :param device: gpu device
    :param optimizer: torch.optim or self defined loss function
    :param criterion: torch.nn or self defined loss function used for optimization
    :param measure: torch.nn or self defined loss function only used for measure
    :param is_train: current status
    :param visualize_step: visualize the training or validation process every n steps
    :param visualize_op: the function used for above visualization
    :param visualize_loss: visualize the each loss during each step
    :return:
    """
    if is_train:
        net.train()
        prefix = ""
        iter = args.epoches_per_phase
    else:
        net.eval()
        prefix = "VAL"
        iter = 1
    # Each step, there will be N types of losses and measures be calculated
    # The structure of the Basic_losses will be:
    # [[loss_1_step_1, loss_2_step_1, ... , loss_N_step_1],
    #   [loss_1_step_2, loss_2_step_2, ..., loss_N_step_2],
    #    ...
    #   [loss_1_step_m, loss_2_step_m, ..., loss_N_step_m]]
    Basic_Losses, Basic_Measures = [], []
    loss_name = args.loss_name
    for epoch in range(iter):
        epoch_loss, epoch_measure = [], []
        start_time = time.time()
        for batch_idx, data in enumerate(dataset):
            if args.steps_per_epoch is not None and batch_idx >= args.steps_per_epoch:
                break
            img_batch, label_batch = data[0].to(device), data[1].to(device)
            prediction, label = net(img_batch, label_batch)
            prediction = [prediction] if type(prediction) is not list else prediction
            label = [label] if type(label) is not list else label
            basic_loss = criterion(prediction, label)
            basic_loss = [basic_loss] if type(basic_loss) is not list else basic_loss
            Basic_Losses.append([float(loss.data) for loss in basic_loss])
            epoch_loss.append(Basic_Losses[-1])
            if measure:
                basic_measure = measure(prediction[-1], label_batch)
                basic_measure = [basic_measure] if type(basic_measure) is not list else basic_measure
                Basic_Measures.append([float(loss.data) for loss in basic_measure])
                epoch_measure.append(Basic_Measures[-1])
            if is_train:
                optimizer.zero_grad()
                total_loss = sum([loss * args.loss_weight[loss_name[i]] for i, loss in enumerate(basic_loss)])
                total_loss.backward()
                optimizer.step()
            # Visualize Output and Loss
            assert len(loss_name) == len(Basic_Losses[-1]), \
                "please check your args.loss_name for it does not match the length of the number of loss produced by the network"
            loss_dict = dict(zip(loss_name, Basic_Losses[-1]))
            if batch_idx % visualize_step == 0:
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
    # Then in the below line , the structure of the Basic_losses
    # and Basic_Measures will be transposed:
    # [[loss_1_step_1, loss_1_step_2, ... , loss_1_step_m],
    #   [loss_2_step_1, loss_2_step_2, ..., loss_2_step_m],
    #    ...
    #   [loss_N_step_1, loss_N_step_2, ..., loss_N_step_m]]
    all_losses = list(zip(*Basic_Losses))
    all_measures = list(zip(*Basic_Measures))
    print(*list(zip(loss_name, [sum(loss) / len(loss) for loss in all_losses])))
    if is_train and visualize_loss:
        vb.visualize_gradient(args, net)
        #args.loss_weight = init.update_loss_weight(all_losses, loss_name, args.loss_weight,
                                                   #args.loss_weight_range, args.loss_weight_momentum)
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