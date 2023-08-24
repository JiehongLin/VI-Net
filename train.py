import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from solver import Solver, get_logger
from dataset import TrainingDataset

def get_parser():
    parser = argparse.ArgumentParser(
        description="VI-Net")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--mode",
                        type=str,
                        default="r",
                        help="[r|ts]")
    parser.add_argument("--checkpoint_epoch",
                        type=int,
                        default=-1,
                        help="checkpoint epoch: -1 / 0")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dataset = args.dataset
    cfg.mode = args.mode
    cfg.gpus = args.gpus
    cfg.checkpoint_epoch = args.checkpoint_epoch
    if cfg.mode == 'ts':
        cfg.log_dir = os.path.join('log', args.dataset, 'PN2')
    elif cfg.mode == 'r':
        cfg.log_dir = os.path.join('log', args.dataset, 'VI_Net')
    else:
        assert False, 'Wrong mode'

    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir("log/"+args.dataset):
        os.makedirs("log/"+args.dataset)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg



if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    if cfg.mode == 'r':
        from VI_Net import Net, Loss
        model = Net(cfg.resolution, cfg.ds_rate)

    else:
        from PN2 import Net, Loss
        model = Net(cfg.n_cls)
    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # loss
    loss = Loss(cfg.loss).cuda()    

    # dataloader
    dataset = TrainingDataset(
        cfg.train_dataset,
        cfg.dataset,
        cfg.mode,
        resolution = cfg.resolution,
        ds_rate = cfg.ds_rate,
        num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.train_dataloader.bs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=int(cfg.train_dataloader.num_workers),
        shuffle=cfg.train_dataloader.shuffle,
        sampler=None,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory
    )

    dataloaders = {
        "train": dataloader,
    }

    # solver
    Trainer = Solver(model=model, loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg)
    Trainer.solve()

    logger.info('\nFinish!\n')
