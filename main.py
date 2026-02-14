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
    t1 = time.time()
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
            #f.write(str(preds.cpu()))
            #f.write('\n')

        val_loss = loss_sum / i_batch
        val_acc = acc_sum / i_batch

    return val_loss, val_acc

def val(model, val_loader, device):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    t1 = time.time()
    with no_grad():
        for i_batch, (data, data_mask, target, target_mask) in enumerate(val_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())

            preds = torch.max(predict_logits, 1)[1].float()
            #f.write(str(preds.cpu()))
            #f.write('\n')

        val_loss = loss_sum / i_batch
        val_acc = acc_sum / i_batch

    return val_loss, val_acc

def train(model, optimizer, train_loader, device):
    t1 = time.time()
    loss_sum = 0
    acc_sum = 0
    model.train()
    for i_batch, (data, data_mask, target, target_mask) in enumerate(train_loader):
        data = data.to(device)
        data_mask = data_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
        acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
        loss_sum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
            t2 = time.time()
            print('Epoch %d, Batch %d, loss = %.4f, acc = %.4f, %.3f seconds/batch' % (
                epoch, i_batch, loss_sum / i_batch, acc_sum / i_batch, (t2 - t1) / config['display_batch_interval']
            ))
            t1 = t2
        '''

    train_loss = loss_sum / i_batch
    train_acc = acc_sum / i_batch

    return train_loss, train_acc



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
    else:
        model = Transformer(d_model=config['d_model'], visual_dim=config['input_video_dim'],
                            target_dim=config['num_class'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(decay_rate=0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    if config['model_load_name'] is not None:
        model.load_state_dict(load(config['model_load_name']))
    model = model.to(device)

    train_loader, val_loader, test_loader = dataloader.creat()

    best_loss = 10
    best_acc = 0.5

    for epoch in range(1, config['max_epoches'] + 1):
        #fres = open(config["results_file"], 'a')
        facc = open(config["acc_file"], 'a')
        #fres.write(str(epoch))
        #fres.write('\n')
        epoch_loss, epoch_acc = train(model, optimizer, train_loader, device)
        epoch_loss_val, epoch_acc_val = val(model, val_loader, device)

        print("epoch: %d train loss: %.4f, train accuracy: %.4f" % (epoch, epoch_loss, epoch_acc))
        print("epoch: %d val loss: %.4f, val accuracy: %.4f" % (epoch, epoch_loss_val, epoch_acc_val))

        facc.write("epoch: %d train loss: %.4f, train accuracy: %.4f" % (epoch, epoch_loss, epoch_acc))
        facc.write('\n')
        facc.write("epoch: %d val loss: %.4f, val accuracy: %.4f" % (epoch, epoch_loss_val, epoch_acc_val))
        facc.write('\n')

        save(model.state_dict(), config["model_save_path"] + 'model-%d.pth' % (epoch))

        epoch_loss_test, epoch_acc_test = test(config["model_save_path"] + 'model-%d.pth' % (epoch), test_loader,
                                               device)
        print("epoch: %d test loss: %.4f, test accuracy: %.4f" % (epoch, epoch_loss_test, epoch_acc_test))
        facc.write("epoch: %d test loss: %.4f, test accuracy: %.4f" % (epoch, epoch_loss_test, epoch_acc_test))
        facc.write('\n')

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            save(model.state_dict(), config["model_save_best_loss"])

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            save(model.state_dict(), config["model_save_best_acc"])

        if epoch % 5 == 0:
            adjust_learning_rate(0.5)

        facc.close()
