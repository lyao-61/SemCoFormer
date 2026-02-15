import os

from Tools.dataloader import dataloader
from Tools.metric import *
from Tools.model import *
from Tools.config import config
from ptflops import get_model_complexity_info


import torch
import torch.nn as nn
torch.manual_seed(3407)
from torch import save, load, no_grad
import time


def test_clip(model_path, test_loader, device):
    model.load_state_dict(torch.load(model_path))
    print("load model from "+ model_path)
    model.eval()
    loss_sum = 0
    acc_sum = 0
    all_preds = []
    all_targets = []
    t1 = time.time()
    with no_grad():
        for i_batch, (data, data_mask, target, target_mask, video_id, obj, subj, frame_list, rel_list) in enumerate(
                test_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            if config["mode"] == "clip_sttran":
                predict_logits, targets, loss = model(
                    data, data_mask, target, target_mask, device,
                    video_id=video_id,
                    obj=obj,
                    subj=subj,
                    frame_list=frame_list,
                    rel_list=rel_list
                )
            else:
                predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            #predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
            all_preds.append(predict_logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            #preds = torch.max(predict_logits, 1)[1].float()
            #print('pred:', preds, 'target:', targets)
            #f.write(str(preds.cpu()))
            #f.write('\n')

        test_loss = loss_sum / i_batch
        test_acc = acc_sum / i_batch
        all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
        all_targets = torch.cat(all_targets, dim=0)  # shape [N]
        test_map = compute_mAP(all_preds, all_targets)
    return test_loss, test_acc, test_map

def val_clip(model, val_loader, device):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    all_preds = []
    all_targets = []
    t1 = time.time()
    with no_grad():
        for i_batch, (data, data_mask, target, target_mask, video_id, obj, subj, frame_list, rel_list) in enumerate(
                val_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            if config["mode"] == "clip_sttran":
                predict_logits, targets, loss = model(
                    data, data_mask, target, target_mask, device,
                    video_id=video_id,
                    obj=obj,
                    subj=subj,
                    frame_list=frame_list,
                    rel_list=rel_list
                )
            else:
                predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            #predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
            all_preds.append(predict_logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            preds = torch.max(predict_logits, 1)[1].float()
            #f.write(str(preds.cpu()))
            #f.write('\n')

        val_loss = loss_sum / i_batch
        val_acc = acc_sum / i_batch
        all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
        all_targets = torch.cat(all_targets, dim=0)  # shape [N]
        val_map = compute_mAP(all_preds, all_targets)

    return val_loss, val_acc, val_map

def train_clip(model, optimizer, train_loader, device):
    t1 = time.time()
    loss_sum = 0
    acc_sum = 0

    all_preds = []
    all_targets = []

    model.train()
    for i_batch, (data, data_mask, target, target_mask, video_id, obj, subj, frame_list, rel_list) in enumerate(train_loader):
    #for i_batch, (data, data_mask, target, target_mask) in enumerate(train_loader):
        data = data.to(device)
        data_mask = data_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)
        if config["mode"]=="clip_sttran":
            predict_logits, targets, loss = model(
                data, data_mask, target, target_mask, device,
                video_id=video_id,
                obj=obj,
                subj=subj,
                frame_list=frame_list,
                rel_list=rel_list
            )
        else:
            predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
        #predict_logits, targets, loss = model(data, data_mask, target, target_mask, device)
        acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
        loss_sum += loss

        all_preds.append(predict_logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
            t2 = time.time()
            print('Epoch %d, Batch %d, loss = %.4f, acc = %.4f, %.3f seconds/batch' % (
                epoch, i_batch, loss_sum / i_batch, acc_sum / i_batch, (t2 - t1) / config['display_batch_interval']
            ))
            t1 = t2

    all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
    all_targets = torch.cat(all_targets, dim=0)  # shape [N]
    train_map = compute_mAP(all_preds, all_targets)

    train_loss = loss_sum / i_batch
    train_acc = acc_sum / i_batch

    return train_loss, train_acc, train_map



#============================================================
def test(model_path, test_loader, device):
    model.load_state_dict(torch.load(model_path))
    print("load model from "+ model_path)
    model.eval()
    loss_sum = 0
    acc_sum = 0
    ap_sum = 0
    all_preds = []
    all_targets = []
    t1 = time.time()

    # 173-201 is prepare for visualize
    # save_path = config["visualize_result_path"]
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #
    # with torch.no_grad():
    #     with open(save_path, "a") as f:
    #         f.write(f"\n\n====== Epoch {epoch} ======\n")
    #         for i_batch, (data, data_mask, target, target_mask) in enumerate(test_loader):
    #             data = data.to(device)
    #             data_mask = data_mask.to(device)
    #             target = target.to(device)
    #             target_mask = target_mask.to(device)
    #
    #             predict_logits, targets, loss = model(data, data_mask, target, target_mask, epoch, device)
    #
    #             loss_sum += loss
    #             acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
    #             all_preds.append(predict_logits.detach().cpu())
    #             all_targets.append(targets.detach().cpu())
    #
    #             preds = torch.argmax(predict_logits, dim=1)
    #
    #             for pred_label, true_label in zip(preds, targets):
    #                 f.write(f"Predicted: {pred_label.item()}, Target: {true_label.item()}\n")
    #
    #     test_loss = loss_sum / i_batch
    #     test_acc = acc_sum / i_batch
    #     all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
    #     all_targets = torch.cat(all_targets, dim=0)  # shape [N]
    #     test_map = compute_mAP(all_preds, all_targets)

    with no_grad():
        for i_batch, (data, data_mask, target, target_mask) in enumerate(test_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            predict_logits, targets, loss = model(data, data_mask, target, target_mask, epoch, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
            all_preds.append(predict_logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

            preds = torch.max(predict_logits, 1)[1].float()
            #ap_sum += compute_mAP(predict_logits.cpu(), target.cpu())
            #print('pred:', preds, 'target:', targets)
            #f.write(str(preds.cpu()))
            #f.write('\n')

        test_loss = loss_sum / i_batch
        test_acc = acc_sum / i_batch
        all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
        all_targets = torch.cat(all_targets, dim=0)  # shape [N]
        test_map = compute_mAP(all_preds, all_targets)

    return test_loss, test_acc, test_map

def val(model, val_loader, device):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    all_preds = []
    all_targets = []

    t1 = time.time()
    with no_grad():
        for i_batch, (data, data_mask, target, target_mask) in enumerate(val_loader):
            data = data.to(device)
            data_mask = data_mask.to(device)
            target = target.to(device)
            target_mask = target_mask.to(device)

            predict_logits, targets, loss = model(data, data_mask, target, target_mask, epoch, device)
            loss_sum += loss
            acc_sum += accuracy(predict_logits.cpu(), targets.cpu())

            all_preds.append(predict_logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

            preds = torch.max(predict_logits, 1)[1].float()
            #f.write(str(preds.cpu()))
            #f.write('\n')

        val_loss = loss_sum / i_batch
        val_acc = acc_sum / i_batch
        all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
        all_targets = torch.cat(all_targets, dim=0)  # shape [N]
        val_map = compute_mAP(all_preds, all_targets)


    return val_loss, val_acc, val_map

def train(model, optimizer, train_loader, device):
    t1 = time.time()
    loss_sum = 0
    acc_sum = 0
    all_preds = []
    all_targets = []

    model.train()
    for i_batch, (data, data_mask, target, target_mask) in enumerate(train_loader):
        data = data.to(device)
        data_mask = data_mask.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        predict_logits, targets, loss = model(data, data_mask, target, target_mask, epoch, device)
        acc_sum += accuracy(predict_logits.cpu(), targets.cpu())
        loss_sum += loss

        all_preds.append(predict_logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i_batch % config['display_batch_interval'] == 0 and i_batch != 0:
            t2 = time.time()
            print('Epoch %d, Batch %d, loss = %.4f, acc = %.4f, %.3f seconds/batch' % (
                epoch, i_batch, loss_sum / i_batch, acc_sum / i_batch, (t2 - t1) / config['display_batch_interval']
            ))
            t1 = t2

    all_preds = torch.cat(all_preds, dim=0)  # shape [N, C]
    all_targets = torch.cat(all_targets, dim=0)  # shape [N]
    train_map = compute_mAP(all_preds, all_targets)

    train_loss = loss_sum / i_batch
    train_acc = acc_sum / i_batch

    return train_loss, train_acc, train_map



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




    elif config["mode"]== "clip_sttran":
        model = Clip_STTran(d_model=config['d_model'], visual_dim=config['input_video_dim'],
                                target_dim=config['num_class'],
                                feat_dim=config['feat_dims'], num_v=config['num_nodes'],
                                dropout=config['dropout_rate'])

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

    best_loss = 10
    best_acc = 0.2
    best_epoch = 1

    for epoch in range(1, config['max_epoches'] + 1):
        #fres = open(config["results_file"], 'a')
        facc = open(config["acc_file"], 'a')
        #fres.write(str(epoch))
        #fres.write('\n')
        if config["mode"]=="clip_sttran":
            epoch_loss, epoch_acc, epoch_map = train_clip(model, optimizer, train_loader, device)
            epoch_loss_val, epoch_acc_val, epoch_map_val = val_clip(model, val_loader, device)

            print("epoch: %d train loss: %.4f, train accuracy: %.4f, train mAp: %.4f" % (
            epoch, epoch_loss, epoch_acc, epoch_map))
            print("epoch: %d val loss: %.4f, val accuracy: %.4f, val mAp: %.4f" % (
            epoch, epoch_loss_val, epoch_acc_val, epoch_map_val))

            facc.write("epoch: %d train loss: %.4f, train accuracy: %.4f, train mAp: %.4f" % (
            epoch, epoch_loss, epoch_acc, epoch_map))
            facc.write('\n')
            facc.write("epoch: %d val loss: %.4f, val accuracy: %.4f, val mAp: %.4f" % (
            epoch, epoch_loss_val, epoch_acc_val, epoch_map_val))
            facc.write('\n')

            save(model.state_dict(), config["model_save_path"] + 'model-%d.pth' % (epoch))

            epoch_loss_test, epoch_acc_test, epoch_map_test = test_clip(config["model_save_path"] + 'model-%d.pth' % (epoch),
                                                                   test_loader, device)
            print("epoch: %d test loss: %.4f, test accuracy: %.4f, test mAP: %.4f" % (
            epoch, epoch_loss_test, epoch_acc_test, epoch_map_test))
            facc.write("epoch: %d test loss: %.4f, test accuracy: %.4f, test mAp: %.4f" % (
            epoch, epoch_loss_test, epoch_acc_test, epoch_map_test))
            facc.write('\n')

            print(config["dataset"], config["mode"])

            if epoch_loss_test < best_loss:
                best_loss = epoch_loss_test
                save(model.state_dict(), config["model_save_best_loss"])

            if epoch_acc_test > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc_test
                save(model.state_dict(), config["model_save_best_acc"])
                facc.write("best epoch: %d test accuracy: %.4f" % (best_epoch, best_acc))
                facc.write('\n')

            if epoch % 5 == 0:
                adjust_learning_rate(0.5)

            facc.close()
        else:
            epoch_loss, epoch_acc, epoch_map = train(model, optimizer, train_loader, device)
            epoch_loss_val, epoch_acc_val, epoch_map_val = val(model, val_loader, device)

            print("epoch: %d train loss: %.4f, train accuracy: %.4f, train map: %.4f" % (epoch, epoch_loss, epoch_acc, epoch_map))
            print("epoch: %d val loss: %.4f, val accuracy: %.4f, val map: %.4f" % (epoch, epoch_loss_val, epoch_acc_val, epoch_map_val))

            facc.write("epoch: %d train loss: %.4f, train accuracy: %.4f, train map: %.4f" % (epoch, epoch_loss, epoch_acc, epoch_map))
            facc.write('\n')
            facc.write("epoch: %d val loss: %.4f, val accuracy: %.4f, val map: %.4f" % (epoch, epoch_loss_val, epoch_acc_val, epoch_map_val))
            facc.write('\n')

            save(model.state_dict(), config["model_save_path"] + 'model-%d.pth' % (epoch))

            epoch_loss_test, epoch_acc_test, epoch_map_test = test(config["model_save_path"] + 'model-%d.pth' % (epoch), test_loader,
                                                   device)
            print("epoch: %d test loss: %.4f, test accuracy: %.4f, test mAP: %.4f" % (epoch, epoch_loss_test, epoch_acc_test, epoch_map_test))
            facc.write("epoch: %d test loss: %.4f, test accuracy: %.4f, test map: %.4f" % (epoch, epoch_loss_test, epoch_acc_test, epoch_map_test))
            facc.write('\n')

            print(config["dataset"], config["mode"])

            if epoch_loss_test < best_loss:
                best_loss = epoch_loss_test
                save(model.state_dict(), config["model_save_best_loss"])

            if epoch_acc_test > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc_test
                save(model.state_dict(), config["model_save_best_acc"])
                facc.write("best epoch: %d test accuracy: %.4f" % (best_epoch, best_acc))
                facc.write('\n')

            if epoch % 5 == 0:
                adjust_learning_rate(0.5)

            facc.close()
