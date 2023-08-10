#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

import torch
from tqdm import tqdm
from common.utils import get_lr
from framework.arg_parser import create_parser
from framework.train_config import TrainConfig
from framework.model_mgr import ModelManager
from framework.train_loop import train_loop
from centernet.loss import focal_loss, reg_l1_loss


    
def epoch_train(model_train, model, loss_history, eval_cb, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler,
                      backbone_name, save_period, save_dir, local_rank):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0

    if local_rank == 0:
        print("Start train")    
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{unfreeze_epoch}', 
                    postfix=dict, mininterval=0.3)
    #reset model_train state.
    model_train.train()
    def forward_hourglass(model_train, batch_imgs, batch_hms, batch_whs,
                         batch_regs, batch_reg_masks):
        outputs = model_train(batch_imgs)
        loss = 0
        c_loss_all = 0
        r_loss_all = 0
        index = 0
        for output in outputs:
            hw, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
            c_loss = focal_loss(hw, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss += c_loss + wh_loss + off_loss
            c_loss_all += c_loss
            r_loss_all += wh_loss + off_loss
            index +=1
        return loss, c_loss_all, r_loss_all, index
    
    def forward_resnet(model_train, batch_imgs, batch_hms, batch_whs, 
                     batch_regs, batch_reg_masks):
        hw, wh, offset = model_train(batch_imgs)
        c_loss = focal_loss(hw, batch_hms)
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

        loss = c_loss + wh_loss + off_loss

        #total_loss += loss.item()
        #total_c_loss += c_loss.item()
        #total_r_loss += wh_loss.item() + off_loss.item()
        return loss, c_loss, wh_loss, off_loss
    
    for iter, batch in enumerate(gen):
        print("iter: ", iter)
        if iter > epoch_step:
            break
        with torch.no_grad():
            if Cuda:
                batch = [ann.cuda(local_rank) for ann  in batch]
        batch_imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        
        optimizer.zero_grad()

        if not fp16:
            print("under fp16 mixed precision")
            if backbone_name == "resnet50":
                loss, c_loss, wh_loss, off_loss = forward_resnet(model_train, batch_imgs, batch_hms, batch_whs,
                             batch_regs, batch_reg_masks)
                # hw, wh, offset = model_train(batch_imgs)
                # c_loss = focal_loss(hw, batch_hms)
                # wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                # off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                # loss = c_loss + wh_loss + off_loss
                print("forward resnet done.")
                total_loss += loss.item()
                total_c_loss += c_loss.item()
                total_r_loss += wh_loss.item() + off_loss.item()
            else:
                loss, c_loss_all, r_loss_all, index = forward_hourglass(model_train, batch_imgs, batch_hms, batch_whs,
                                batch_regs, batch_reg_masks)
                total_loss += loss.item() / index
                total_c_loss += c_loss_all.item() / index
                total_r_loss += r_loss_all.item() / index
        else:
            from torch.cuda.amp import autocast  as autocast
            with autocast():
                if backbone_name == "resnet50":
                    loss, c_loss, wh_loss, off_loss = forward_resnet(model_train, batch_imgs, batch_hms, batch_whs,
                             batch_regs, batch_reg_masks)
                
                    total_loss += loss.item()
                    total_c_loss += c_loss.item()
                    total_r_loss += wh_loss.item() + off_loss.item()
                else:
                    loss, c_loss_all, r_loss_all, index = forward_hourglass(model_train, batch_imgs, batch_hms, batch_whs,
                                batch_regs, batch_reg_masks)
                    total_loss += loss.item() / index
                    total_c_loss += c_loss_all.item() / index
                    total_r_loss += r_loss_all.item() / index
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iter + 1),
                                'total_c_loss': total_c_loss / (iter + 1),
                                'lr' : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print("Finish train")
        print("start val")
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{unfreeze_epoch}',postfix=dict, mininterval=.3)

    model_train.eval()
    for iter, batch in enumerate(gen_val):
        if iter >= epoch_step_val:
            break

        with torch.no_grad():
            if Cuda:
                batch = [ann.cuda(local_rank) for ann  in batch]
            batch_imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            if backbone_name == "resnet50":
                loss, _,  _, _ = forward_resnet(model_train, batch_imgs, batch_hms, batch_whs,
                             batch_regs, batch_reg_masks)
                val_loss += loss.item()
            else:
                loss, _, _, index = forward_hourglass(model_train, batch_imgs, batch_hms, batch_whs,
                                batch_regs, batch_reg_masks)
                val_los += loss.item() / index
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iter + 1)})
                pbar.update(1)
        
                
                


    # end of this epoch
    if local_rank == 0:
        pbar.close()
        print("Finish vald")
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/epoch_step_val)
        eval_cb.on_epoch_end(epoch + 1, model_train)
        print("Epoch:" + str(epoch + 1) + "/" + str(unfreeze_epoch))
        print("Total Loss: %.3f || Val loss: %.3f " % (total_loss / epoch_step, val_loss/epoch_step_val))
        
        # save weight
        if (epoch + 1) % save_period == 0  or epoch + 1 == unfreeze_epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'centernet_epoch_' + str(epoch + 1) + '.pth'))

        if len(loss_history.val_loss)  <= 1 or (val_loss / epoch_step_val) < min(loss_history.val_loss[:-1]):
            print("save best model")
            torch.save(model.state_dict(), os.path.join(save_dir, 'centernet_best.pth'))

        torch.save(model.state_dict(), os.path.join(save_dir, 'centernet_last.pth'))




def train():
    flags = create_parser()
    args = flags.parse_args()
    print(args)
   
    tc = TrainConfig(args)
    model_mgr = ModelManager(tc)

    model_mgr.init_backbone()
    model = model_mgr.get_model()
    model_train = tc.get_model_train(model)

    #print(model)
    train_loop(model, model_train,  tc, epoch_train)
    


if __name__ == "__main__":
    train()