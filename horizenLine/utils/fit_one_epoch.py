"""
训练单轮
"""
import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from horizenLine.utils.common_utils import vis_train_samples
from ultralytics import solutions


def fit_one_epoch(model, model_name, scales, train_data, val_data, train_epoch_step, val_epoch_step, epoch,
                  epochs, vis_train_sample_flag, optimizer, loss_function, loss_history, device, model_save_dir, mlr,
                  lr, Freeze_Train):
    # 记录单轮loss
    loss_train, loss_train_1, loss_train_2 = list(), list(), list()
    loss_val = list()
    # 训练过程
    if mlr:
        if Freeze_Train:
            if epochs == 50:
                for p in optimizer.param_groups:
                    p['lr'] = lr * 0.99 ** epoch
            elif epochs == 100:
                for p in optimizer.param_groups:
                    p['lr'] = lr * 0.99 ** (epoch - 50)
        else:
            for p in optimizer.param_groups:
                p['lr'] = lr * 0.99 ** epoch

    print("[train] {}/{}, lr:{}".format(str(epoch + 1), str(epochs), str(optimizer.param_groups[0]["lr"])))
    # model.train()
    with tqdm(total=train_epoch_step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for i, data in tqdm(enumerate(train_data)):
            imgs, heatmaps = data  # 都是经过数据增强的
            if vis_train_sample_flag:
                vis_train_samples(imgs, heatmaps[0], scales)

            img = imgs.to(device)
            heatmap_vps_1 = heatmaps[0].to(device)
            if model_name == "yolo11":
                heatmap_vps_2 = heatmaps[1].to(device)

                # 前向传播 ,
                # output_1, output_2 = model(img)
                # l_train_1 = loss_function(output_1, heatmap_vps_1)
                # l_train_2 = loss_function(output_2, heatmap_vps_2)
                # l_train = l_train_1 + l_train_2
                heatmap = solutions.Heatmap(
                    show=True,
                    model="yolo11n-pose.pt",
                    colormap=cv2.COLORMAP_PARULA,
                )
                output_1 = model(img)
                im0 = heatmap.generate_heatmap(output_1)
                l_train_1 = loss_function(im0, heatmap_vps_1)
                l_train = l_train_1

                loss_train_1.append(l_train_1.item())
                # loss_train_2.append(l_train_2.item())
                loss_train.append(l_train.item())

                pbar.set_postfix(**{"out0_loss": np.mean(np.array(loss_train_1)),
                                    "out1_loss": np.mean(np.array(loss_train_2)),
                                    "total_train_loss": np.mean(np.array(loss_train))})
                pbar.update(1)
            elif model_name.split("-")[0] == "resnet":
                output = model(img)
                l_train = loss_function(output, heatmap_vps_1)
                loss_train.append(l_train.item())

                pbar.set_postfix(**{"total_train_loss": np.mean(np.array(loss_train))})
                pbar.update(1)

            # 优化
            optimizer.zero_grad()
            l_train.backward()
            optimizer.step()

    # 验证过程
    print("[val] {}/{}".format(str(epoch + 1), str(epochs)))
    model.eval()
    with tqdm(total=val_epoch_step, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for i, data in enumerate(val_data):
            imgs, heatmaps = data

            img = imgs.to(device)
            heatmap_vps_1 = heatmaps[0].to(device)
            if model_name == "yolov11":
                heatmap_vps_2 = heatmaps[1].to(device)

                # output_1, output_2 = model(img)
                # l_val_1 = loss_function(output_1, heatmap_vps_1)
                # l_val_2 = loss_function(output_2, heatmap_vps_2)
                #
                # l_val = l_val_1 + l_val_2
                #
                # loss_val.append(l_val.item())

                heatmap = solutions.Heatmap(
                    show=True,
                    model="yolo11n-pose.pt",
                    colormap=cv2.COLORMAP_PARULA,
                )

                im0 = heatmap.generate_heatmap(img)
                l_val_1 = loss_function(im0, heatmap_vps_1)
                l_val = l_val_1

                loss_val.append(l_val.item())

                pbar.set_postfix(**{"total_val_loss": np.mean(np.array(loss_val))})
                pbar.update(1)
            elif model_name.split("-")[0] == "resnet":
                output = model(img)
                l_val = loss_function(output, heatmap_vps_1)
                loss_val.append(l_val.item())

                pbar.set_postfix(**{"total_val_loss": np.mean(np.array(loss_val))})
                pbar.update(1)

    loss_history.append_loss(epoch + 1, np.mean(np.array(loss_train)), np.mean(np.array(loss_val)))
    model_save_path = os.path.join(model_save_dir,
                                   "epoch{}-{}-loss-{:.12f}-val_loss-{:.12f}.pth".format(str(epoch + 1),
                                                                                         str(model_name),
                                                                                         np.mean(np.array(loss_train)),
                                                                                         np.mean(np.array(loss_val))))
    torch.save(model.state_dict(), model_save_path)
    print("epoch: {} train_loss: {} val_loss: {}".format(str(epoch + 1), str(np.mean(np.array(loss_train))),
                                                         str(np.mean(np.array(loss_val)))))
    return np.mean(np.array(loss_val))
