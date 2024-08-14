
import os
import time
import logging
from tqdm import tqdm
import pickle as cPickle
import numpy as np
import torch
import torch.optim as optim
import numpy as np
import gorilla
from tensorboardX import SummaryWriter
from common_utils import write_obj
from scheduler import BNMomentumScheduler #, CyclicLR
import torch.nn.functional as F
from evaluation_utils import compute_3d_matches_for_each_gt
from vis_utils import draw_detections

class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, data_mode, loss, dataloaders, logger, cfg, start_epoch=1, start_iter=0):
        super(Solver, self).__init__(
            model=model,
            dataloaders=dataloaders,
            cfg=cfg,
            logger=logger,
        )
        self.loss = loss
        self.data_mode = data_mode
        self.logger.propagate = 0

        tb_writer_ = tools_writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer = tb_writer_

        self.per_val = cfg.per_val
        self.per_write = cfg.per_write
        self.epoch = start_epoch
        self.iter = start_iter
        if cfg.get("freeze_world_enhancer", False):
            # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

        self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, max_lr=1e-3,
                                            step_size_up=cfg.max_epoch * cfg.num_mini_batch_per_epoch // 6, mode='triangular', cycle_momentum=False)
        # self.weihtdecay = optim.lr_scheduler.CosineAnnealingWarmRestarts()

        bnm_lmbd = lambda it: max(cfg.bn.bn_momentum*cfg.bn.bn_decay**(int(it / cfg.bn.decay_step)), cfg.bn.bnm_clip)
        self.bnm_scheduler = BNMomentumScheduler(self.model, bn_lambda=bnm_lmbd, last_epoch=self.iter)

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

            if self.epoch % 2 == 0:
                ckpt_path = os.path.join(
                    self.cfg.log_dir, 'epoch_' + str(self.epoch) + '.pth')
                gorilla.solver.save_checkpoint(
                    model=self.model, filename=ckpt_path, optimizer=self.optimizer, meta={'iter': self.iter, "epoch": self.epoch})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()

        self.dataloaders["syn"].dataset.reset()
        self.dataloaders["real"].dataset.reset()

        i=0

        for syn_data, real_data in zip(self.dataloaders["syn"], self.dataloaders["real"]):
            data_time = time.time()-end

            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step(self.iter)

            # if self.bnm_scheduler is not None:
            #     self.bnm_scheduler.step(self.iter)

            self.optimizer.zero_grad()
            loss, dict_info_step = self.step(syn_data, real_data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()
            backward_time = time.time() - end - forward_time-data_time
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.iter)

            if self.bnm_scheduler is not None:
                self.bnm_scheduler.step(self.iter)

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
                    self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["syn"]), self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()

            self.iter += 1
            i+=1

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

    def step(self, syn_data, real_data, mode):
        torch.cuda.synchronize()
        b1 = syn_data['rgb'].size(0)
        b2 = real_data['rgb'].size(0)
        n1 = syn_data['pts'].size(1)

        for key in syn_data:
            syn_data[key] = syn_data[key].cuda()
        for key in real_data:
            real_data[key] = real_data[key].cuda()

        data = {
            'rgb': torch.cat([syn_data['rgb'], real_data['rgb']], dim=0),
            'pts': torch.cat([syn_data['pts'], real_data['pts']], dim=0),
            'choose': torch.cat([syn_data['choose'], real_data['choose']], dim=0),
            'category_label': torch.cat([syn_data['category_label'], real_data['category_label']], dim=0),
            'model':  torch.cat([syn_data['model'], real_data['model']], dim=0),
            'sym_info': torch.cat([syn_data['sym_info'], real_data['sym_info']], dim=0),
            'gt_R': torch.cat([syn_data['rotation_label'], real_data['rotation_label']], dim=0),
            'gt_t': torch.cat([syn_data['translation_label'], real_data['translation_label']], dim=0),
            'gt_s': torch.cat([syn_data['size_label'], real_data['size_label']], dim=0),
            'qo': torch.cat([syn_data['qo'], real_data['qo']], dim=0),
        }
        end_points = self.model(data)
        for key in end_points:
            syn_data[key] = end_points[key][0:b1]
            real_data[key] = end_points[key][b1:]

        loss_syn = self.loss['syn'](syn_data)
        loss_real = self.loss['real'](real_data)
        loss_all = (loss_syn * b1 + loss_real * b2) / (b1+b2)
        dict_info = {
            'loss_all': float(loss_all.item()),
            'loss_syn': float(loss_syn.item()),
            'loss_real': float(loss_real.item())
        }

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_last_lr()[0]
        return loss_all, dict_info


    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
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


def test_func(model, dataloader, save_path):
    model.eval()
    inference_times = []

    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):
            # Process inputs for the model
            inputs = {
                'rgb': data['rgb'][0].cuda(),
                'pts': data['pts'][0].cuda(),
                'choose': data['choose'][0].cuda(),
                'category_label': data['category_label'][0].cuda(),
            }

            # Measure inference time
            start_time = time.time()
            end_points = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            pred_translation = end_points['pred_translation']
            pred_size = end_points['pred_size']
            pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
            pred_size = pred_size / pred_scale
            pred_rotation = end_points['pred_rotation']

            num_instance = pred_rotation.size(0)
            pred_RTs = torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
            pred_RTs[:, :3, 3] = pred_translation
            pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
            pred_scales = pred_size

            # Save results
            result = {
                'gt_class_ids': data['gt_class_ids'][0].numpy(),
                'gt_bboxes': data['gt_bboxes'][0].numpy(),
                'gt_RTs': data['gt_RTs'][0].numpy(),
                'gt_scales': data['gt_scales'][0].numpy(),
                'gt_handle_visibility': data['gt_handle_visibility'][0].numpy(),
                'pred_class_ids': data['pred_class_ids'][0].numpy(),
                'pred_bboxes': data['pred_bboxes'][0].numpy(),
                'pred_RTs': pred_RTs.detach().cpu().numpy(),
                'pred_scales': pred_scales.detach().cpu().numpy()
            }

            save_filename = os.path.join(save_path, f'result_{i}.pkl')
            with open(save_filename, 'wb') as f:
                cPickle.dump(result, f)

            # Optional: Visualize the box
            draw_box_to_image(data, result, "real", i)

            t.set_description(
                f"Test [{i+1}/{len(dataloader)}][{num_instance}]: "
            )
            t.update(1)

    # Calculate and log inference speed
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time per batch: {average_inference_time:.4f} seconds")
    print(f"Average inference speed: {1.0 / average_inference_time:.4f} FPS")

def draw_box_to_image(data, result, data_name, img_id):
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    out_dir = "/home/midea1/Desktop/IST-Net/log/ist_net_default/img"

    img = data['ori_img'][0].numpy()
    gt_class_ids = result['gt_class_ids']
    gt_bboxes = result['gt_bboxes']
    gt_RTs = result['gt_RTs']
    gt_scales = result['gt_scales']
    gt_handle_visibility = result['gt_handle_visibility']
    pred_class_ids = result['pred_class_ids']
    pred_bboxes = result['pred_bboxes']
    pred_scores = result.get('pred_scores', np.zeros(len(pred_class_ids)))

    pred_RTs = result['pred_RTs']
    pred_scales = result['pred_scales']

    iou_cls_gt_match, iou_pred_indices = compute_3d_matches_for_each_gt(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                                                                        pred_bboxes, pred_class_ids, pred_scores, pred_RTs, pred_scales)

    pred_class_ids = pred_class_ids[iou_pred_indices]
    pred_RTs = pred_RTs[iou_pred_indices]
    pred_scores = pred_scores[iou_pred_indices]
    pred_bboxes = pred_bboxes[iou_pred_indices]

    pred_RTs = pred_RTs[iou_cls_gt_match]
    pred_scales = pred_scales[iou_cls_gt_match]
    pred_class_ids = pred_class_ids[iou_cls_gt_match]

    draw_detections(img, out_dir, "real", img_id, intrinsics, pred_RTs, pred_scales, pred_class_ids,
                    gt_RTs, gt_scales, gt_class_ids, None, None, None, draw_gt=True, draw_nocs=False)




class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
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

