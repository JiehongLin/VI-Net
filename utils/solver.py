import os
import time
import logging
from tqdm import tqdm
import pickle as cPickle

import torch
import gorilla
from tensorboardX import SummaryWriter

class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg):
        super(Solver, self).__init__(
            model=model,
            dataloaders=dataloaders,
            cfg=cfg,
            logger=logger,
        )
        self.loss = loss
        self.logger.propagate = 0

        tb_writer_ = tools_writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer = tb_writer_

        self.per_val = cfg.per_val
        self.per_write = cfg.per_write

        if cfg.checkpoint_epoch != -1:
            logger.info("=> loading checkpoint from epoch {} ...".format(cfg.checkpoint_epoch))
            checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.checkpoint_epoch) + '.pth')
            checkpoint_file = gorilla.solver.resume(model=model, filename=checkpoint, optimizer=self.optimizer, scheduler=self.lr_scheduler)
            start_epoch = checkpoint_file['epoch']+1
            start_iter = checkpoint_file['iter']
            del checkpoint_file
        else:
            start_epoch = 1
            start_iter = 0
        self.epoch = start_epoch
        self.iter = start_iter


    def solve(self):
        while self.epoch <= self.cfg.max_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value

            ckpt_path = os.path.join(
                self.cfg.log_dir, 'epoch_0.pth')
            gorilla.solver.save_checkpoint(
                model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={'iter': self.iter, "epoch": self.epoch})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()
        self.dataloaders["train"].dataset.reset()

        for i, data in enumerate(self.dataloaders["train"]):

            data_time = time.time()-end

            self.optimizer.zero_grad()
            loss, dict_info_step = self.step(data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()
            backward_time = time.time() - end - forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                # ipdb.set_trace()
                self.log_buffer.average(self.per_write)
                prefix = '[{}/{}][{}/{}][{}] Train - '.format(
                    self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["train"]), self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.iter += 1


        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def evaluate(self):
        mode = 'eval'
        self.model.eval()

        for i, data in enumerate(self.dataloaders["eval"]):
            with torch.no_grad():
                _, dict_info_step = self.step(data, mode)
                self.log_buffer.update(dict_info_step)
                if i % self.per_write == 0:
                    self.log_buffer.average(self.per_write)
                    prefix = '[{}/{}][{}/{}] Test - '.format(
                        self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["eval"]))
                    write_info = self.get_logger_info(
                        prefix, dict_info=self.log_buffer._output)
                    self.logger.info(write_info)
                    self.write_summary(self.log_buffer._output, mode)
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def step(self, data, mode):
        torch.cuda.synchronize()
        for key in data:
            data[key] = data[key].cuda()
        end_points = self.model(data)
        dict_info = self.loss(end_points, data)
        loss_all = dict_info['loss']

        for key in dict_info:
            dict_info[key] = float(dict_info[key].item())

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_last_lr()[0]

        return loss_all, dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            elif 'lr' in key:
                info = info + '{}: {:.6f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False


def test_func(ts_model, r_model, dataloder, save_path):
    ts_model.eval()
    r_model.eval()
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            path = dataloder.dataset.result_pkl_list[i]

            # save
            result = {}

            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()

            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()

            result['gt_scales'] = data['gt_scales'][0].numpy()
            result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()

            result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            result['pred_scores'] = data['pred_scores'][0].numpy()

            ### prediction

            if 'pts' in data.keys():
                center = data['center'][0].cuda()


                inputs = {
                    'rgb': data['rgb'][0].cuda(),
                    'pts': data['pts'][0].cuda(),
                    'category_label': data['category_label'][0].cuda(),
                }
                end_points = ts_model(inputs)
                pred_translation = inputs['translation'] = end_points['translation']
                pred_size = inputs['size'] = end_points['size']
                pred_scale = torch.norm(pred_size, dim=1, keepdim=True)

                pts = (inputs['pts'] - pred_translation.unsqueeze(1))/ (pred_scale + 1e-8).unsqueeze(2)
                inputs['pts'] = pts.detach()

                end_points = r_model(inputs)
                pred_rotation = end_points['pred_rotation'][:,:,(1,2,0)]


                pred_size = pred_size / pred_scale

                num_instance = pred_rotation.size(0)
                pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
                pred_RTs[:, :3, 3] = pred_translation + center
                pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
                pred_scales = pred_size

                result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
                result['pred_scales'] = pred_scales.detach().cpu().numpy()
                with open(os.path.join(save_path, path.split('/')[-1]), 'wb') as f:
                    cPickle.dump(result, f)

            else:
                import numpy as np
                ninstance = data['pred_class_ids'][0].numpy().shape[0]
                result['pred_RTs'] = np.zeros((ninstance, 4, 4))
                result['pred_RTs'][:, :3, :3] = np.diag(np.ones(3))
                result['pred_scales'] = np.ones((ninstance, 3))


            draw = False
            if draw:
                index = data['index'].item()
                path = dataloder.dataset.result_pkl_list[index]
                with open(path, 'rb') as f:
                    data = cPickle.load(f)
                image_path = data['image_path'][5:]
                image_path = os.path.join(
                    '/cluster/public_datasets/SemGroup/NOCS', image_path)
                image_path_parsing = image_path.split('/')

                # rgb
                import cv2
                import numpy as np
                from draw_utils import draw_detections

                image = cv2.imread(image_path + '_color.png')[:, :, :3]
                image = image[:, :, ::-1] #480*640*3
                intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


                if not os.path.isdir(os.path.join(save_path, 'draw')):
                    os.mkdir(os.path.join(save_path, 'draw'))
                draw_detections(image, os.path.join(save_path, 'draw'), 'real_test', image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics,
                                    result['gt_bboxes'], None, None, result['gt_RTs'], result['gt_scales'],
                                    result['pred_bboxes'], result['pred_class_ids'], None, result['pred_RTs'], None, result['pred_scales'],
                                    draw_gt=False, draw_pred=True)


            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, len(dataloder), num_instance)
            )

            t.update(1)




class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        # writer = SummaryWriter(dir_project)
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger

