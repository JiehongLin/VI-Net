import os
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data_utils import (
    load_depth,
    load_composed_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,
    random_rotate,
    random_scale,
)

class TrainingDataset(Dataset):
    def __init__(self,
            config, 
            dataset='REAL275',
            mode='ts',
            num_img_per_epoch=-1,
            resolution=64,
            ds_rate=2
    ):
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.num_img_per_epoch = num_img_per_epoch

        self.resolution = resolution
        self.ds_rate = ds_rate
        self.sample_num = self.config.sample_num
        self.data_dir = config.data_dir

        syn_img_path = 'camera/train_list.txt'
        self.syn_intrinsics = [577.5, 577.5, 319.5, 239.5]
        self.syn_img_list = [os.path.join(syn_img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, syn_img_path))]
        print('{} synthetic images are found.'.format(len(self.syn_img_list)))

        if self.dataset == 'REAL275':
            real_img_path = 'real/train_list.txt'
            self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
            self.real_img_list = [os.path.join(real_img_path.split('/')[0], line.rstrip('\n'))
                            for line in open(os.path.join(self.data_dir, real_img_path))]
            print('{} real images are found.'.format(len(self.real_img_list)))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        if self.num_img_per_epoch != -1:
            self.reset()

    def __len__(self):
        if self.num_img_per_epoch == -1:
            if self.dataset == 'REAL275':
                return len(self.syn_img_list) + len(self.real_img_list)
            else:
                return len(self.syn_img_list)
        else:
            return self.num_img_per_epoch
        
    def reset(self):
        assert self.num_img_per_epoch != -1
        if self.dataset == 'REAL275':
            num_syn_img = len(self.syn_img_list)
            num_syn_img, num_real_img = len(self.syn_img_list), len(self.real_img_list)
            num_syn_img_per_epoch = int(self.num_img_per_epoch*0.75)
            num_real_img_per_epoch = self.num_img_per_epoch - num_syn_img_per_epoch

            if num_syn_img <= num_syn_img_per_epoch:
                syn_img_index = np.random.choice(num_syn_img, num_syn_img_per_epoch)
            else:
                syn_img_index = np.random.choice(num_syn_img, num_syn_img_per_epoch, replace=False)

            if num_real_img <= num_real_img_per_epoch:
                real_img_index = np.random.choice(num_real_img, num_real_img_per_epoch)
            else:
                real_img_index = np.random.choice(num_real_img, num_real_img_per_epoch, replace=False)
            real_img_index = -real_img_index - 1
            self.img_index = np.hstack([syn_img_index, real_img_index])

        else:
            num_syn_img = len(self.syn_img_list)
            num_syn_img_per_epoch = int(self.num_img_per_epoch)
            if num_syn_img <= num_syn_img_per_epoch:
                syn_img_index = np.random.choice(num_syn_img, num_syn_img_per_epoch)
            else:
                syn_img_index = np.random.choice(num_syn_img, num_syn_img_per_epoch, replace=False)
            self.img_index = syn_img_index

        np.random.shuffle(self.img_index)

    def __getitem__(self, index):
        while True:
            image_index = self.img_index[index]
            data_dict = self._read_data(image_index)
            if data_dict is None:
                index = np.random.randint(self.__len__())
                continue
            return data_dict
        
    def _read_data(self, image_index):
        if image_index>=0:
            img_type = 'syn'
            img_path = os.path.join(self.data_dir, self.syn_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
        else:
            img_type = 'real'
            image_index = -image_index-1
            img_path = os.path.join(self.data_dir, self.real_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        if self.dataset == 'REAL275':
            depth = load_composed_depth(img_path)
            depth = fill_missing(depth, self.norm_scale, 1)
        else:
            depth = load_depth(img_path)

        # mask
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        num_instance = len(gts['instance_ids'])
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640

        idx = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        mask = np.equal(mask, gts['instance_ids'][idx])
        mask = np.logical_and(mask , depth > 0)

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            return None
        elif len(choose) <= self.sample_num:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        pts2 = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1))[choose] / self.norm_scale
        pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32) # 480*640*3
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb[rmin:rmax, cmin:cmax, :]
    
        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
        rgb = np.array(rgb)
        if img_type == 'syn':
            rgb = rgb_add_noise(rgb)
        rgb = rgb.astype(np.float32).reshape((-1,3))[choose, :] / 255.0

        # gt
        translation = gts['translations'][idx].astype(np.float32)
        rotation = gts['rotations'][idx].astype(np.float32)
        size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)


        if hasattr(self.config, 'random_rotate') and self.config.random_rotate:
            pts, rotation = random_rotate(pts, rotation, translation, self.config.angle_range)

        if self.mode == 'ts':
            pts, size = random_scale(pts, size, rotation, translation)

            center = np.mean(pts, axis=0)
            pts = pts - center[np.newaxis, :]
            translation = translation - center

            noise_t = np.random.uniform(-0.02, 0.02, 3)
            pts = pts + noise_t[None, :]
            translation = translation + noise_t

            ret_dict = {}
            ret_dict['pts'] = torch.FloatTensor(pts)
            ret_dict['rgb'] = torch.FloatTensor(rgb)
            ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
            ret_dict['translation_label'] = torch.FloatTensor(translation)
            ret_dict['size_label'] = torch.FloatTensor(size)

        else:
            noise_t = np.random.uniform(-0.02, 0.02, 3)
            noise_s = np.random.uniform(0.8, 1.2, 1)
            pts = pts - translation[None, :] - noise_t[None, :]
            pts = pts / np.linalg.norm(size) * noise_s

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map

                asym_flag = 0.0
            else:
                asym_flag = 1.0

            # transform ZXY system to XYZ system
            rotation = rotation[:, (2,0,1)]

            v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
            rho = np.arctan2(v[1], v[0])
            if v[1]<0:
                rho += 2*np.pi
            phi = np.arccos(v[2])

            vp_rotation = np.array([
                [np.cos(rho),-np.sin(rho),0],
                [np.sin(rho), np.cos(rho),0],
                [0,0,1]
            ]) @ np.array([
                [np.cos(phi),0,np.sin(phi)],
                [0,1,0],
                [-np.sin(phi),0,np.cos(phi)],
            ])
            ip_rotation = vp_rotation.T @ rotation

            rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
            phi_label = int(phi/np.pi*(self.resolution//self.ds_rate))

            ret_dict = {}
            ret_dict['rgb'] = torch.FloatTensor(rgb)
            ret_dict['pts'] = torch.FloatTensor(pts)
            ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
            ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
            ret_dict['translation_label'] = torch.FloatTensor(translation)
            ret_dict['rotation_label'] = torch.FloatTensor(rotation)
            ret_dict['size_label'] = torch.FloatTensor(size)

            ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
            ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
            ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
            ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)

        return ret_dict
    


class TestDataset():
    def __init__(self, config, dataset='REAL275', resolution=64):
        self.dataset = dataset
        self.resolution = resolution
        self.data_dir = config.data_dir
        self.sample_num = config.sample_num

        result_pkl_list = glob.glob(os.path.join(self.data_dir, 'segmentation_results', dataset, 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        if dataset == 'REAL275':
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]


    def __len__(self):
        return len(self.result_pkl_list)

    def __getitem__(self, index):
        path = self.result_pkl_list[index]

        with open(path, 'rb') as f:
            pred_data = cPickle.load(f)

        image_path = os.path.join(self.data_dir, pred_data['image_path'][5:])
        pred_mask = pred_data['pred_masks']
        num_instance = len(pred_data['pred_class_ids'])

        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        depth = load_depth(image_path) #480*640
        if self.dataset == 'REAL275':
            depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3

        all_rgb = []
        all_pts = []
        all_center = []
        all_cat_ids = []
        flag_instance = torch.zeros(num_instance) == 1

        for j in range(num_instance):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            if np.sum(mask) > 16:
                rmin, rmax, cmin, cmax = get_bbox_from_mask(mask)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed

                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :].copy()
                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = np.array(instance_rgb).astype(np.float32).reshape((-1, 3))[choose, :] / 255.0

                center = np.mean(instance_pts, axis=0)
                instance_pts = instance_pts - center[np.newaxis, :]

                if instance_pts.shape[0] <= self.sample_num:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num)
                else:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num, replace=False)
                instance_pts = instance_pts[choose_idx, :]
                instance_rgb = instance_rgb[choose_idx, :]

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_center.append(torch.FloatTensor(center))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                flag_instance[j] = 1

        ret_dict = {}

        ret_dict['gt_class_ids'] = torch.tensor(pred_data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(pred_data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(pred_data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(pred_data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(pred_data['gt_handle_visibility'])
        ret_dict['index'] = index

        if len(all_pts) == 0:
            ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])
            ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])
            ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])

        else:
            ret_dict['pts'] = torch.stack(all_pts) # N*3
            ret_dict['rgb'] = torch.stack(all_rgb)
            ret_dict['center'] = torch.stack(all_center)
            ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
            ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
            ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
            ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]

        return ret_dict
