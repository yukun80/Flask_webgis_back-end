import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, Combined_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score

from torch.utils.tensorboard import SummaryWriter


def fit_one_epoch(
    model_train,
    model,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    Lovasz_Loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank,
    TrainName,
    log_dir,
):
    writer = SummaryWriter(log_dir=log_dir)
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            elif Lovasz_Loss:
                # 使用交叉熵loss与lovasz loss结合
                loss = Combined_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                elif Lovasz_Loss:
                    # 使用交叉熵loss与lovasz loss结合
                    loss = Combined_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "total_loss": total_loss / (iteration + 1),
                    "f_score": total_f_score / (iteration + 1),
                    "lr": get_lr(optimizer),
                }
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Train")
        print("Start Validation")
        pbar = tqdm(total=epoch_step_val, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(
                    **{
                        "val_loss": val_loss / (iteration + 1),
                        "f_score": val_f_score / (iteration + 1),
                        "lr": get_lr(optimizer),
                    }
                )
                pbar.update(1)

    if local_rank == 0:
        TrainName += "model"
        if not os.path.exists(os.path.join(save_dir, TrainName)):
            os.makedirs(os.path.join(save_dir, TrainName))
        
        pbar.close()
        print("Finish Validation")
        # 在log_dir下创建一个名为TrainName的文件夹
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))
        print("Total Loss: %.3f || Val Loss: %.3f " % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    TrainName,
                    "ep%03d-loss%.3f-val_loss%.3f.pth"
                    % ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val),
                ),
            )

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print("Save best model to best_epoch_weights.pth")
            # 保存最优模型，文件名为时间+模型名+epoch
            torch.save(model.state_dict(), os.path.join(save_dir, TrainName, "best_epoch_weights.pth"))
            # torch.save(model.state_dict(), os.path.join(save_dir, TrainName,  "best_epoch_weights.pth"))

        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    # 在训练和验证循环结束后，将 f_score 写入 TensorBoard
    writer.add_scalar("train_f_score", total_f_score / epoch_step, epoch)
    writer.add_scalar("val_f_score", val_f_score / epoch_step_val, epoch)
