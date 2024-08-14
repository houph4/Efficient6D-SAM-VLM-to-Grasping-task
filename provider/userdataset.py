import os
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
from data.dataset_wild6d import Wild6DDataset
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from data_augmentation import data_augment, get_rotation
import cv2
import numpy as np
from pyk4a import PyK4A, Config
import pyk4a
from ultralytics import FastSAM

from utils.data_utils import get_bbox, fill_missing


class TestDataset(Dataset):
    def __init__(self, images, config):
        self.images = images
        self.img_size = config.img_size
        self.sample_num = config.sample_num
        self.intrinsics = [913.0894775390625, 912.6361694335938, 960.84765625, 551.1499633789062]

        self.xmap = np.array([[i for i in range(1920)] for j in range(1080)])
        self.ymap = np.array([[j for i in range(1920)] for j in range(1080)])
        self.sym_ids = [0, 1, 3]
        self.norm_scale = 1000.0
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.class_name_map = {
            0: 'real',
            1: 'bottle_',
            2: 'bowl_',
            3: 'camera_',
            4: 'can_',
            5: 'laptop_',
            6: 'mug_'
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_image, depth_image = self.images[index]

        model = FastSAM('kinect/FastSAM-s.pt')
        print('FastSAM model loaded.')

        rgb = rgb_image[:, :, ::-1]
        points = np.array([[366, 854], [1113, 850], [1100, 400], [327, 404]])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        cropped_rgb_image = rgb_image[y_min:y_max, x_min:x_max]
        results = model.predict(cropped_rgb_image, conf=0.85, show=True, save=True)
        print(results)
        pred_mask = results[0].masks.data.cpu().numpy()
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        depth = fill_missing(depth_image, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1, 2, 0)).astype(np.float32)

        all_rgb = []
        all_pts = []
        all_cat_ids = []
        all_choose = []

        flag_instance = torch.zeros(num_instance=1) == 1

        for j in range(num_instance=1):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(bboxes[:4])
            
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) > 16:
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                cat_id = 3

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1

        ret_dict = {
            'pts': torch.stack(all_pts),
            'rgb': torch.stack(all_rgb),
            'ori_img': torch.tensor(rgb_image),
            'choose': torch.stack(all_choose),
            'category_label': torch.stack(all_cat_ids).squeeze(1),
            'pred_class_ids': torch.tensor(4)[flag_instance == 1],
            'pred_bboxes': torch.tensor(bboxes)[flag_instance == 1]
        }

        return ret_dict
