import torch
from tqdm import tqdm
from common.utils import get_lr
from framework.arg_parser import create_parser
from framework.train_config import TrainConfig
from framework.model_mgr import ModelManager
from framework.train_loop import train_loop


def epoch_train(model_train, model, loss_history, eval_cb, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, Cuda, fp16, scaler,
                      backbone_name, save_period, save_dir, local_rank):
    pass

def train():
    flags = create_parser()
    args = flags.parse_args()
    
    args.model_name = "ssd"
    args.backbone = "resnet50"
    args.dataset = "voc_2007"
    print(args)
    tc = TrainConfig(args)
    model_mgr = ModelManager(tc)

    

    
    # model_mgr.init_backbone()
    # model = model_mgr.get_model()
    # model_train = tc.get_model_train(model)


    # #print(model)
    # train_loop(model, model_train,  tc, epoch_train)
    


if __name__ == "__main__":
    train()