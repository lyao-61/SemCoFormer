from Tools.dataloader import dataloader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
if __name__ == "__main__":

    train_iter, val_iter, test_iter = dataloader.creat()

    for i_batch, (data, data_mask, target, target_mask) in enumerate(val_iter):

        print(data.size())
        print(target.size())