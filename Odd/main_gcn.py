from Tools.dataloader import iterator_factory_gcn as iterator_factory
from Tools.metric import *
from Tools.metric1 import accuracy
from Tools.model import *
from Tools.config import config


import torch
import torch.nn as nn
from torch import save, load, no_grad
import time


def val(model, val_loader):
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

            ##predict_logits, targets, loss = model(data, data_mask, target, target_mask)
            ##loss_sum += loss
            ##acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
            logit = model(data.float())
            target = torch.topk(target, 1, dim=2)[1]
            target = torch.squeeze(target)
            acc_sum += accuracy(logit, target[:, -1])
            loss = criterion(logit, target[:, -1])
            loss_sum += loss.item()

            if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
                t2 = time.time()
                print('Val: Batch %d, loss = %.4f, acc = %.4f, %.3f seconds/batch' % (
                    i_batch, loss_sum / i_batch, acc_sum / i_batch, (t2 - t1) / config['display_batch_interval']
                ))
                t1 = t2
        val_loss = loss_sum / i_batch
        val_acc = acc_sum / i_batch

    return val_loss, val_acc

def train(model, optimizer, train_loader):
    t1 = time.time()
    loss_sum = 0
    acc_sum = 0
    model.train()
    for i_batch, (data, data_mask, target, target_mask) in enumerate(train_loader):
        data = data.to(device)
        data_mask = data_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        ##predict_logits, targets, loss = model(data, data_mask, target, target_mask)
        ##acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
        ##loss_sum += loss

        logit = model(data.float())
        target = torch.topk(target, 1, dim=2)[1]
        target = torch.squeeze(target)
        acc_sum += accuracy(logit, target[:, -1])
        loss = criterion(logit, target[:, -1])
        loss_sum += loss.item()

        #predict_probs = F.softmax(predict_logits, -1)
        #predicts = torch.argmax(predict_probs, dim=1)

        #acc = Accuracy()
        #acc.update(preds=[predict_logits.cpu()], labels=[targets.cpu()])
        #acc_sum += acc.get()[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
            t2 = time.time()
            print('Epoch %d, Batch %d, loss = %.4f, acc = %.4f, %.3f seconds/batch' % (
                epoch, i_batch, loss_sum / i_batch, acc_sum / i_batch, (t2 - t1) / config['display_batch_interval']
            ))
            t1 = t2

    train_loss = loss_sum / i_batch
    train_acc = acc_sum / i_batch

    return train_loss, train_acc



if __name__ == '__main__':

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    A = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    A = torch.from_numpy(np.asarray(A)).to(device)

    model = GGCN(A, config['num_nodes'], config['num_class'], [config['num_nodes'], 128],
                 [128, 256, 256, 1024], config['feat_dims'], config['dropout_rate'])

    '''
    model = Locator1(d_model=config['d_model'],
                    frame_dim=config['input_video_dim'],
                    word_dim=config['num_class'])'''

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    def adjust_learning_rate(decay_rate=0.8):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    if config['model_load_name'] is not None:
        model.load_state_dict(load(config['model_load_name']))
    model = model.to(device)

    train_loader, val_loader = iterator_factory.creat(file_num=config['video_num'], ratio=config['ratio'], batch_size=config["batch_size"])

    best_loss = 10
    for epoch in range(1, config['max_epoches'] + 1):
        epoch_loss, epoch_acc = train(model, optimizer, train_loader)
        epoch_loss_val, epoch_acc_val = val(model, val_loader)
        print("epoch: %d train loss: %.4f, train accuracy: %.4f" % (epoch, epoch_loss, epoch_acc))
        print("epoch: %d val loss: %.4f, val accuracy: %.4f" % (epoch, epoch_loss_val, epoch_acc_val))

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            save(model.state_dict(), config['model_save_best'])
            print(config['dataset'], config['mode'], config['model_save_best'])
        save(model.state_dict(), config['model_save_last'])

        if epoch % 5 == 0:
            adjust_learning_rate(0.5)