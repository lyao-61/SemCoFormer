from Tools.dataloader import dataloader
from Tools.metric import *
from Tools.model import *
from Tools.config import config


import torch
import torch.nn as nn
from torch import save, load, no_grad
import time


def test(model_path, test_loader, device):
    model.load_state_dict(torch.load(model_path))
    print("load model from "+ model_path)
    model.eval()
    loss_sum = 0
    acc_sum = 0
    ap_sum = 0
    with no_grad():
        for i_batch, (data, data_mask, target, target_mask) in enumerate(test_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())

            preds = torch.max(predict_logits, 1)[1].float()
            ap_sum += compute_mAP(predict_logits.cpu(), targets.cpu())
            if (accuracy(predict_logits.cpu(), targets.cpu())<0.4):
                print(i_batch+1)
                print(preds.cpu().numpy().astype(np.int8))
                print(targets.cpu().numpy())

        test_loss = loss_sum / (i_batch+1)
        test_acc = acc_sum / (i_batch+1)
        test_ap = ap_sum / (i_batch+1)

    return test_loss, test_acc, test_ap

if __name__ == '__main__':

    device = torch.device(config['cuda'] if torch.cuda.is_available() else 'cpu')

    if config["mode"]== "gcn_transformer":
        A = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
        A = torch.from_numpy(np.asarray(A)).to(device)

        model = GCN_Transformer(d_model=config['d_model'], visual_dim=config['input_video_dim'],
                                target_dim=config['num_class'],
                                feat_dim=config['feat_dims'], adj_matix=A, num_v=config['num_nodes'],
                                dropout=config['dropout_rate'])
    elif config["mode"]== "gcn":
        A = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
        A = torch.from_numpy(np.asarray(A)).to(device)

        model = GGCN(visual_dim=config['input_video_dim'],
                     target_dim=config['num_class'],
                     feat_dim=config['feat_dims'], adj_matix=A, num_v=config['num_nodes'],
                     dropout=config['dropout_rate'])
    elif config["mode"]== "transformer":
        model = Transformer(d_model=config['d_model'], visual_dim=config['input_video_dim'],
                            target_dim=config['num_class'])

    else:
        model = RNNModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(decay_rate=0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    if config['model_load_name'] is not None:
        model.load_state_dict(load(config['model_load_name']))
    model = model.to(device)

    train_loader, val_loader, test_loader = dataloader.creat()

    epoch = 21

    # fres = open(config["results_file"], 'a')
    facc = open(config["acc_test_file"], 'a')

    epoch_loss_test, epoch_acc_test, epoch_ap_test = test(config["model_save_path"] + 'model-%d.pth' % (epoch),
                                                          test_loader,
                                                          device)
    print("epoch: %d test loss: %.4f, test accuracy: %.4f, test mAP: %.4f" % (
    epoch, epoch_loss_test, epoch_acc_test, epoch_ap_test))
    facc.write("epoch: %d test loss: %.4f, test accuracy: %.4f" % (epoch, epoch_loss_test, epoch_acc_test))
    facc.write('\n')

    print(config["dataset"], config["mode"])

    facc.close()
