import cv2
import math
import json
import torch
import pickle
import logging
import numpy as np
import pandas as pd
import numpy.matlib
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.nn import init
from skimage import filters
from torchvision import transforms
from medpy.metric import binary

import framework.logger
from models.utils import all_score, soft_argmax_tensor, shape_to_mask, shape_to_dice_tensor
from models.ASM import ActiveShapeModel, pts_transform_np

torch.backends.cudnn.benchmark = True
logger = logging.getLogger()


def config_log(log_path):
    '''
    重定向log保存位置
    '''
    log_name = log_path.joinpath('log.log')
    framework.logger.config_logger(log_name)
    logger = logging.getLogger()
    return logger


def gaussian(pt, H, W, sigmma=1, gamma=7):
    '''
    根据pt位置，计算HW的mask各点高斯值，其中sigmma可以设定为可学习的参数。
    '''
    pt_W, pt_H = pt
    pt_Ws = np.matlib.repmat(pt_W, H, W)
    pt_Hs = np.matlib.repmat(pt_H, H, W)
    map_x = np.matlib.repmat(np.arange(W), H, 1)
    map_y = np.matlib.repmat(np.arange(H), W, 1)
    map_y = np.transpose(map_y)

    dist = np.sqrt((map_x - pt_Ws) ** 2 + (map_y - pt_Hs) ** 2)
    gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist / (sigmma**2))
    # cv2.imwrite('gaussian.png', gaussian_map*255)
    return gaussian_map


def make_pts_heatmap_np(shape, WH, sigmma=3, gamma=7):
    '''
    制作点的热力图
    shape: [N, 2]
    '''
    np.seterr(divide='ignore', invalid='ignore')
    masks = np.zeros((shape.shape[0], WH[1], WH[0]), dtype=np.float32)
    for i in range(shape.shape[0]):
        pt = shape[i].astype(int)
        mask = gaussian(pt, WH[1], WH[0], sigmma, gamma)
        mask = mask / mask.max()
        masks[i] = mask
    return masks


class Trainer_DSM:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda') if config.gpu else torch.device('cpu')
        self.Loss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()
        self.dataloader = self.config.dataloader(self.config)
        self.test_loader = self.dataloader.load_data('test', self.config.val_batch_size)
        self.proj_path = Path(config.save_path)
        self.input_size = torch.Tensor(config.input_size).to(self.device)
        self.sasm, self.masms = self.load_models()
        self.model_save_dir = self.proj_path.joinpath('model', config.train_mode)
        self.vis_train_dir = self.proj_path.joinpath('vis_train')
        for d in [self.model_save_dir, self.vis_train_dir]:
            if not d.exists():
                d.mkdir(parents=True)

    def vis_key_pts(self, image, pts_coord_pred, pts, pt_idxes):
        '''
        将pts_pred和pts画在image上
        inputs are all tensor
        '''
        image = image.numpy() * 255
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 0, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 0, 128],
        ]
        for i in range(len(pt_idxes)):
            cv2.circle(image_color, (int(pts_coord_pred[i][0]), int(pts_coord_pred[i][1])), 3, colors[i], 5)
            cv2.circle(image_color, (int(pts[i][0]), int(pts[i][1])), 3, colors[i], 5)
        return image_color

    def vis_all_pts(self, image, all_pts_coord_pred, vis_idx=True, label=None):
        image = image.numpy() * 255
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if label is not None:
            label = label.astype(np.uint8) * 255  # .transpose(1, 2, 0)
            edge = cv2.Canny(label, 30, 100)
            contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(image_color, contour, -1, (0, 255, 0), 2)
        for i in range(all_pts_coord_pred.shape[0]):
            pt = all_pts_coord_pred[i]
            cv2.circle(image_color, (int(pt[0]), int(pt[1])), 2, [0, 0, 255], 2)
            if vis_idx:
                cv2.putText(
                    image_color, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1
                )
        return image_color

    def vis_pred_boundary(self, image, label, mask, dice=None):
        image = image.numpy() * 255
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # 画label边缘
        label = label.astype(np.uint8) * 255  # .transpose(1, 2, 0)
        edge = cv2.Canny(label, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image_color, contour, -1, (0, 255, 0), 2)  # 绿色
        # 画mask边缘
        edge = cv2.Canny(mask * 255, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image_color, contour, -1, (0, 0, 255), 2)  # 绿色
        # 写上dice
        if dice is not None:
            cv2.putText(
                image_color, 'dice: {:.4f}'.format(dice), (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
            )
        return image_color

    def vis_pred_boundary_and_asm(self, image, label, seg_mask, asm_mask, dice=None):
        image = image.numpy() * 255
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # 画label边缘
        label = label.astype(np.uint8) * 255  # .transpose(1, 2, 0)
        edge = cv2.Canny(label, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image_color, contour, -1, (0, 255, 0), 2)  # 绿色
        # 画seg_mask边缘
        edge = cv2.Canny(seg_mask, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image_color, contour, -1, (0, 255, 255), 2)  # 绿色
        # 画masm_mask边缘
        edge = cv2.Canny(asm_mask * 255, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image_color, contour, -1, (0, 0, 255), 2)  # 黄色
        # 写上dice
        if dice is not None:
            cv2.putText(
                image_color, 'dice: {:.4f}'.format(dice), (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
            )
        return image_color

    def init_shape(self, key_pts_hm_pred, key_pts_coord_pred, pt_idxes, seg_pred, shape_pts_num):
        '''
        根据key landmark，将mean shape映射过来，并生成全部点的heatmap，其中key landmark的heatmap沿用前面预测的
        key_pts_hm_pred: [B, k, H, W]
        key_pts_coord_pred: [B, k, 2]
        seg_pred: 用来筛选masm，[B, 1, H, W]，stem直接预测的cuda tensor
        return: shape [B, N, 2] and all_pts_hm [B, N, H, W]
        '''
        seg_pred_cpu = seg_pred.detach().cpu()
        # threshold = filters.threshold_otsu(seg_pred_cpu.numpy())
        seg_pred_cpu_binary = torch.where(
            seg_pred_cpu > 0.5, torch.ones_like(seg_pred_cpu), torch.zeros_like(seg_pred_cpu)
        )
        B, K, H, W = key_pts_hm_pred.shape
        key_pts_coord_pred = key_pts_coord_pred.detach().cpu().numpy()
        all_pts_hm = torch.zeros((B, shape_pts_num, H, W))  # [B, N, H, W]
        sigmma = int(self.config.heatmap_dir_name.split('_')[-1])
        shapes = []
        for sample_idx in range(B):  # 遍历每张sample
            best_dice = 0
            best_masm = self.sasm
            for masm in self.masms:  # 为每张sample选择最优masm
                shape = pts_transform_np(key_pts_coord_pred[0], pt_idxes, masm.mean_shape.cpu().numpy())
                shapes.append(shape)
                dice = shape_to_dice_tensor(shape, seg_pred_cpu_binary[sample_idx])
                if dice > best_dice:
                    best_dice = dice
                    best_masm = masm
            # 择其最优的masm，生成全部点的heatmap
            shape = pts_transform_np(
                key_pts_coord_pred[sample_idx], pt_idxes, best_masm.mean_shape.cpu().numpy()
            )  # [N, 2]
            hms = make_pts_heatmap_np(shape, self.config.input_size, sigmma)  # [N, H, W]
            hms = torch.from_numpy(hms)
            all_pts_hm[sample_idx] = hms
        all_pts_hm = all_pts_hm.to(self.device)
        for i in range(len(pt_idxes)):
            key_pt_idx = pt_idxes[i]
            all_pts_hm[:, key_pt_idx] = key_pts_hm_pred[:, i]
        return shape, all_pts_hm, shapes

    def load_models(self):
        '''
        加载masm的k个model，以及sasm的一个model
        '''
        asm_dir = Path(self.config.asm_dir)
        sasm = pickle.load(open(asm_dir.joinpath('model'), 'rb')).to(self.device)
        masms = []
        for idx in range(self.config.k):
            masm_path = asm_dir.parent.joinpath('{}_clusters'.format(self.config.k), str(idx), 'model')
            masm = pickle.load(open(masm_path, 'rb')).to(self.device)
            masms.append(masm)
        return sasm, masms

    def init_shape_coord(self, key_pts_coord_pred):
        '''
        根据key landmark，将mean shape映射过来，选择最优的shape，
        生成全部点的heatmap，其中key landmark的heatmap沿用前面预测的
        key_pts_coord_pred: [B, k, 2]
        '''
        B, K, D = key_pts_coord_pred.shape
        key_pts_coord_pred_np = key_pts_coord_pred.detach().cpu().numpy()
        initial_shapes = torch.zeros((B, self.config.total_pts_num, D))
        for sample_idx in range(B):
            shape = pts_transform_np(key_pts_coord_pred_np[sample_idx], self.config.pt_idxes, self.mean_shape_np)
            shape = torch.from_numpy(shape)  # [N, 2]
            initial_shapes[sample_idx] = shape
        initial_shapes = initial_shapes.to(self.device)
        initial_shapes[:, self.config.pt_idxes] = key_pts_coord_pred
        return initial_shapes

    def evaluate_hd_assd(self, seg_pred, label):
        seg_pred_ = seg_pred.squeeze(0).detach().cpu().numpy()
        threshold = filters.threshold_otsu(seg_pred_)
        seg_pred_binary = np.where(seg_pred_ > threshold, 255.0, 0.0).squeeze()
        label = label.squeeze().numpy()
        hd95 = binary.hd95(seg_pred_binary, label)
        assd = binary.assd(seg_pred_binary, label)
        return seg_pred_binary, hd95, assd

    def train_stem(self):
        '''
        先训练stem
        '''
        dataloader = self.dataloader.load_data(self.config.train_mode, self.config.train_batch_size)
        vis_dir = self.vis_train_dir.joinpath(self.config.train_mode, 'stem')
        model_save_dir = self.model_save_dir.joinpath('stem')
        for d in [vis_dir, model_save_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        logger = config_log(vis_dir)
        minLoss = 1e12
        bestDice = 0
        stem = self.config.stem(1, self.config.channels, self.config.stem_out_ch).to(self.device)
        stem.train()
        optimizer = torch.optim.Adam(stem.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        for epoch in tqdm(range(1, self.config.epochs + 1)):
            for image, label, all_pts_hm, shape, b, image_name in dataloader:
                image, label, all_pts_hm = image.to(self.device), label.to(self.device), all_pts_hm.to(self.device)
                # shape, b = shape.to(self.device), b.to(self.device)
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                # calculate loss
                seg_loss = self.Loss(seg_pred, label)
                key_pts_hm_loss = self.Loss(key_pts_hm_pred, all_pts_hm[:, self.config.pt_idxes])
                loss = seg_loss + key_pts_hm_loss * 10
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break
            scheduler.step()
            if epoch % self.config.step == 0:
                evaluation, val_losses = self.evaluate_stem(stem, vis_dir)
                acc, auc, sen, spe, pre, iou, dc = evaluation
                logger.info(
                    '{}:\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'.format(
                        epoch, acc, auc, sen, spe, pre, iou, dc
                    )
                )
                logger.info('{}:\t {:.8f}\t {:.8f}\t {:.4f}'.format(epoch, val_losses[0], val_losses[1], val_losses[2]))
                if sum(val_losses) < minLoss:
                    minLoss = sum(val_losses)
                    torch.save(stem, model_save_dir.joinpath('stem_minLoss'))
                    logger.info('Saving minLoss......')
                if dc > bestDice:
                    bestDice = dc
                    torch.save(stem, model_save_dir.joinpath('stem_bestDice'.format(bestDice)))
                    torch.save(stem, model_save_dir.joinpath('stem_{:.4f}'.format(bestDice)))
                    logger.info('Saving bestDice......')
        # 保存一个最终模型
        torch.save(stem, model_save_dir.joinpath('stem_final'))

    def evaluate_stem(self, model_lst, vis_dir=None):
        stem = model_lst
        stem.eval()
        seg_loss = 0
        key_pts_hm_loss = 0
        key_pts_coord_loss = 0
        evaluation = np.array([0.0 for _ in range(7)])
        with torch.no_grad():
            for image, label, all_pts_hm, shape, b, image_name in self.test_loader:
                image = image.to(self.device)
                image_name = image_name[0]
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                # calculate loss
                seg_loss += self.Loss(seg_pred.detach().cpu(), label)
                # seg_pred取argmax
                seg_pred = torch.max(seg_pred, dim=1, keepdim=True).values
                label = torch.max(label, dim=1, keepdim=True).values
                temp = all_score(seg_pred.squeeze().detach().cpu().numpy(), label.squeeze().numpy())
                # seg_pred_sum = torch.sum(seg_pred, dim=1, keepdim=True)
                # label_sum = torch.sum(label, dim=1, keepdim=True)
                # temp = all_score(seg_pred_sum.detach().cpu().numpy(), label_sum.numpy())
                evaluation += temp
                key_pts_hm_loss = self.Loss(key_pts_hm_pred.detach().cpu(), all_pts_hm[:, self.config.pt_idxes])
                key_pts_coord_loss = self.Loss(key_pts_coord_pred.detach().cpu(), shape[:, self.config.pt_idxes]) / 1e2
                if vis_dir:
                    mask = self.vis_key_pts(
                        image.cpu().squeeze(),
                        key_pts_coord_pred.squeeze() * self.config.resize_ratio,
                        shape[:, self.config.pt_idxes].squeeze() * self.config.resize_ratio,
                        self.config.pt_idxes,
                    )
                    cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_key.png')), mask)
                    seg_pred_img = transforms.ToPILImage()(seg_pred.squeeze(0).cpu())
                    seg_pred_img.save(vis_dir.joinpath(image_name[:-4] + '_seg.png'))
                    for j in range(key_pts_hm_pred.shape[1]):
                        pt_img = key_pts_hm_pred[0, j]
                        pt_img = transforms.ToPILImage()(pt_img)
                        pt_idx = self.config.pt_idxes[j]
                        pt_img.save(vis_dir.joinpath('{}_{}.png'.format(image_name[:-4], pt_idx)))
                # break
        stem.train()
        num = len(self.test_loader)
        losses = [seg_loss / num, key_pts_hm_loss / num, key_pts_coord_loss / num]
        return evaluation / (num * self.config.val_batch_size), losses

    def infer_stem(self):
        '''
        可以指定其他model，因为stem_minLoss的结果已经在 vis_train/stem里了
        实际上用不到，因为fineturn以后才是最终模型
        '''
        save_dir = self.proj_path.joinpath('infer', self.config.train_mode, 'stem', self.config.infer_name)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        stem = torch.load(self.model_save_dir.joinpath('stem', 'stem_' + self.config.infer_name)).to(self.device)
        cols = ['name', 'acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']
        stem_seg_results = pd.DataFrame(columns=cols)  # 保存每张图的结果
        i = 0
        for image, label, all_pts_hm, shape, b, image_name in tqdm(self.test_loader):
            image_name = image_name[0]
            # stem_seg_results.at[i, 'name'] = image_name
            stem_seg_result = [image_name]
            image = image.to(self.device)
            with torch.no_grad():
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
            if self.config.dataset_name == 'ms':
                seg_pred_L_binary, hd95_L, assd_L = self.evaluate_hd_assd(seg_pred[:, 0], label[:, 0])
                seg_pred_R_binary, hd95_R, assd_R = self.evaluate_hd_assd(seg_pred[:, 1], label[:, 1])
                seg_pred_binary = seg_pred_L_binary + seg_pred_R_binary
                hd95 = (hd95_L + hd95_R) / 2
                assd = (assd_L + assd_R) / 2
            else:
                seg_pred_binary, hd95, assd = self.evaluate_hd_assd(seg_pred, label)
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg_binary.png')), seg_pred_binary)
            temp = all_score(torch.sum(seg_pred.cpu(), 1).numpy(), np.sum(label.numpy(), 1))
            stem_seg_result = stem_seg_result + list(temp) + [hd95, assd]
            stem_seg_results.loc[i] = stem_seg_result
            i += 1
            key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
            # for vis
            mask = self.vis_key_pts(
                image.cpu().squeeze(),
                key_pts_coord_pred.squeeze() * self.config.resize_ratio,
                shape[:, self.config.pt_idxes].squeeze() * self.config.resize_ratio,
                self.config.pt_idxes,
            )
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_key.png')), mask)
            # seg_pred_img = transforms.ToPILImage()(torch.sum(seg_pred, 1).squeeze(0).cpu())
            # seg_pred_img.save(save_dir.joinpath(image_name[:-4] + '_seg.png'))
            for j in range(key_pts_hm_pred.shape[1]):
                pt_img = key_pts_hm_pred[0, j]
                pt_img = transforms.ToPILImage()(pt_img)
                pt_idx = self.config.pt_idxes[j]
                pt_img.save(save_dir.joinpath('{}_{}.png'.format(image_name[:-4], pt_idx)))
            # break
        # 保存数值结果
        mean = stem_seg_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].mean()
        result = ['mean'] + list(mean)
        stem_seg_results.loc[i] = result
        i += 1
        std = stem_seg_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].std()
        result = ['std'] + list(std)
        stem_seg_results.loc[i] = result
        stem_seg_results.set_index(['name'], inplace=True)
        stem_seg_results.to_csv(save_dir.joinpath('stem_seg_results.csv'))

    def train_left(self):
        '''
        加载stem后，训练其余部分
        '''
        dataloader = self.dataloader.load_data(self.config.train_mode, self.config.train_batch_size)
        vis_dir = self.vis_train_dir.joinpath(self.config.train_mode, 'left')
        model_save_dir = self.model_save_dir.joinpath('left')
        for d in [vis_dir, model_save_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        logger = config_log(vis_dir)
        minLoss = 1e12
        bestDice = 0.0
        stem = torch.load(self.model_save_dir.joinpath('stem', 'stem_minLoss')).to(self.device)
        D12 = self.config.d4_net(self.config.channels, False, 1, self.config.decoder_conv).to(self.device)
        E2 = self.config.e4_net(
            self.config.channels[1] + self.config.total_pts_num + self.config.stem_out_ch[0],
            self.config.channels,
            self.config.decoder_conv,
        ).to(self.device)
        D2 = self.config.d4_net(self.config.channels, True, self.config.total_pts_num, self.config.decoder_conv).to(
            self.device
        )
        stem.eval()
        D12.train()
        E2.train()
        D2.train()
        optimizer = torch.optim.Adam(
            [{'params': D12.parameters()}, {'params': E2.parameters()}, {'params': D2.parameters()}],
            lr=self.config.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        for epoch in tqdm(range(1, self.config.epochs + 1)):
            for image, label, all_pts_hm, shape, b, image_name in dataloader:
                image, label, all_pts_hm = image.to(self.device), label.to(self.device), all_pts_hm.to(self.device)
                shape, b = shape.to(self.device), b.to(self.device)
                with torch.no_grad():
                    seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                if self.config.dataset_name == 'ms':
                    initial_shape_L, initial_shape_hm_L = self.init_shape(
                        key_pts_hm_pred[:, :4],
                        key_pts_coord_pred[:, :4],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 0],
                        26,
                    )
                    initial_shape_R, initial_shape_hm_R = self.init_shape(
                        key_pts_hm_pred[:, 4:],
                        key_pts_coord_pred[:, 4:],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 1],
                        26,
                    )
                    initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
                else:  # CAMUS
                    initial_shape, initial_shape_hm = self.init_shape(
                        key_pts_hm_pred,
                        key_pts_coord_pred,
                        self.config.pt_idxes,
                        half_seg_pred,
                        self.config.total_pts_num,
                    )
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
                if torch.isnan(all_pts_hm_pred).any():
                    print('train NaN:', image_name)
                    break
                # all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
                # bs_pred = []
                # for idx in range(image.shape[0]):
                #     if self.config.dataset_name == 'ms':
                #         b_pred_L, _ = self.sasm.transform(all_pts_coord_pred[idx, :26])
                #         b_pred_R, _ = self.sasm.transform(all_pts_coord_pred[idx, 26:])
                #         b_pred = torch.cat((b_pred_L, b_pred_R), dim=0)
                #     else:
                #         b_pred, _ = self.sasm.transform(all_pts_coord_pred[idx])  # [10]
                #     bs_pred.append(b_pred)
                # bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
                # calculate loss
                # seg_loss = self.Loss(seg_pred, label) * 1e2
                all_pts_hm_loss = self.Loss(all_pts_hm_pred, all_pts_hm)
                if epoch > self.config.epochs // 2:
                    # b_loss = self.Loss(bs_pred, b)
                    loss = all_pts_hm_loss  # + b_loss
                else:
                    loss = all_pts_hm_loss
                optimizer.zero_grad()
                # if not torch.isnan(all_pts_hm_pred).any():
                loss.backward()
                optimizer.step()
                # break
            scheduler.step()

            if epoch % self.config.snapshot == 0:
                torch.save(D12, model_save_dir.joinpath('D12_{}'.format(epoch)))
                torch.save(E2, model_save_dir.joinpath('E2_{}'.format(epoch)))
                torch.save(D2, model_save_dir.joinpath('D2_{}'.format(epoch)))

            if epoch % self.config.step == 0:
                evaluation, val_losses = self.evaluate_left([stem, D12, E2, D2], vis_dir)
                acc, auc, sen, spe, pre, iou, dc = evaluation
                logger.info(
                    '{}:\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'.format(
                        epoch, acc, auc, sen, spe, pre, iou, dc
                    )
                )
                logger.info(
                    '{}:\t {:.8f}\t {:.8f}\t {:.4f}\t {:.8f}\t {:.4f}, {:.4f}'.format(
                        epoch, val_losses[0], val_losses[1], val_losses[2], val_losses[3], val_losses[4], val_losses[5]
                    )
                )
                if sum(val_losses) < minLoss:
                    minLoss = sum(val_losses)
                    torch.save(D12, model_save_dir.joinpath('D12_minLoss'))
                    torch.save(E2, model_save_dir.joinpath('E2_minLoss'))
                    torch.save(D2, model_save_dir.joinpath('D2_minLoss'))
                    logger.info('Saving minLoss......')
                if dc >= bestDice:
                    bestDice = dc
                    torch.save(D12, model_save_dir.joinpath('D12_bestDice'))
                    torch.save(E2, model_save_dir.joinpath('E2_bestDice'))
                    torch.save(D2, model_save_dir.joinpath('D2_bestDice'))
                    # torch.save(D12, model_save_dir.joinpath('D12_{:.4f}'.format(bestDice)))
                    # torch.save(E2, model_save_dir.joinpath('E2_{:.4f}'.format(bestDice)))
                    # torch.save(D2, model_save_dir.joinpath('D2_{:.4f}'.format(bestDice)))
                    logger.info('Saving bestDice......')
        # 保存一个最终模型
        torch.save(D12, model_save_dir.joinpath('D12_final'))
        torch.save(E2, model_save_dir.joinpath('E2_final'))
        torch.save(D2, model_save_dir.joinpath('D2_final'))

    def evaluate_left(self, model_lst, vis_dir=None):
        stem, D12, E2, D2 = model_lst
        D12.eval()
        E2.eval()
        D2.eval()
        seg_loss = 0
        key_pts_hm_loss = 0
        key_pts_coord_loss = 0
        all_pts_hm_loss = 0.0
        all_pts_coord_loss = 0.0
        b_loss = 0.0
        evaluation = np.array([0.0 for _ in range(7)])
        for image, label, all_pts_hm, shape, b, image_name in self.test_loader:
            image_name = image_name[0]
            # if image_name == 'QG22052901-L>42_G.png':
            #     continue
            # print(image_name)
            image = image.to(self.device)
            label = torch.max(label, dim=1, keepdim=True).values
            with torch.no_grad():
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                if self.config.dataset_name == 'ms':
                    initial_shape_L, initial_shape_hm_L = self.init_shape(
                        key_pts_hm_pred[:, :4],
                        key_pts_coord_pred[:, :4],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 0],
                        26,
                    )
                    initial_shape_R, initial_shape_hm_R = self.init_shape(
                        key_pts_hm_pred[:, 4:],
                        key_pts_coord_pred[:, 4:],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 1],
                        26,
                    )
                    initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
                else:  # CAMUS
                    initial_shape, initial_shape_hm = self.init_shape(
                        key_pts_hm_pred,
                        key_pts_coord_pred,
                        self.config.pt_idxes,
                        half_seg_pred,
                        self.config.total_pts_num,
                    )
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
            # all_pts_hm_pred = all_pts_hm_pred.detach().cpu()
            # 判断all_pts_hm_pred是否有nan，有则跳过
            # if torch.isnan(all_pts_hm_pred).any():
            #     print('eval NaN:', image_name)
            #     continue
            all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
            seg_pred = torch.max(seg_pred.detach().cpu(), dim=1, keepdim=True).values
            bs_pred = []
            for idx in range(image.shape[0]):
                # b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[idx])  # [10]
                if self.config.dataset_name == 'ms':
                    b_pred_L, asm_shape_L = self.sasm.transform(all_pts_coord_pred[idx, :26])
                    b_pred_R, asm_shape_R = self.sasm.transform(all_pts_coord_pred[idx, 26:])
                    b_pred = torch.cat((b_pred_L, b_pred_R), dim=0)
                    # asm_shape = torch.cat((asm_shape_L, asm_shape_R), dim=0)
                    shape_mask_L_np = shape_to_mask(asm_shape_L.cpu(), self.config.input_size)
                    shape_mask_L_np = cv2.resize(shape_mask_L_np, (label.shape[-1], label.shape[-2]))
                    shape_mask_R_np = shape_to_mask(asm_shape_R.cpu(), self.config.input_size)
                    shape_mask_R_np = cv2.resize(shape_mask_R_np, (label.shape[-1], label.shape[-2]))
                    shape_mask_np = shape_mask_L_np + shape_mask_R_np
                    # 将init shape可视化
                    init_shape_mask_L_np = shape_to_mask(initial_shape_L, self.config.input_size)
                    init_shape_mask_L_np = cv2.resize(init_shape_mask_L_np, (label.shape[-1], label.shape[-2]))
                    init_shape_mask_R_np = shape_to_mask(initial_shape_R, self.config.input_size)
                    init_shape_mask_R_np = cv2.resize(init_shape_mask_R_np, (label.shape[-1], label.shape[-2]))
                    init_shape_mask_np = init_shape_mask_L_np + init_shape_mask_R_np
                else:
                    b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[idx])
                    shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
                    shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
                bs_pred.append(b_pred)
                # 得到shape，与label计算分割指标
                temp = all_score(shape_mask_np, label.squeeze().numpy())
                evaluation += temp
                if vis_dir:
                    image = image.cpu().squeeze()
                    pts_vis = self.vis_all_pts(image, all_pts_coord_pred[idx] * self.config.resize_ratio)
                    cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_pts.png')), pts_vis)
                    init_vis = self.vis_pred_boundary(image, label.squeeze().numpy(), init_shape_mask_np)
                    cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_init.png')), init_vis)
                    seg_vis = self.vis_pred_boundary(image, label.squeeze().numpy(), shape_mask_np, temp[-1])
                    cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_asm.png')), seg_vis)
                    seg_pred_ = transforms.ToPILImage()(seg_pred.squeeze(0))
                    seg_pred_.save(vis_dir.joinpath(image_name[:-4] + '_seg.png'))
            bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
            # calculate loss
            seg_loss += self.Loss(seg_pred, label)
            key_pts_hm_loss = self.Loss(key_pts_hm_pred.detach().cpu(), all_pts_hm[:, self.config.pt_idxes])
            key_pts_coord_loss = self.Loss(key_pts_coord_pred.detach().cpu(), shape[:, self.config.pt_idxes])
            all_pts_hm_loss = self.Loss(all_pts_hm_pred.detach().cpu(), all_pts_hm)
            all_pts_coord_loss = self.Loss(all_pts_coord_pred.detach().cpu(), shape)
            b_loss = self.Loss(bs_pred.detach().cpu(), b)
            # break
        D12.train()
        E2.train()
        D2.train()
        num = len(self.test_loader)
        losses = [
            seg_loss / num,
            key_pts_hm_loss / num,
            key_pts_coord_loss / num,
            all_pts_hm_loss / num,
            all_pts_coord_loss / num,
            b_loss / num,
        ]
        return evaluation / (num * self.config.val_batch_size), losses

    def infer_whole(self, data_loader, save_dir):
        '''
        infer整个流程，保存多个中间结果，包括stem、final
        '''
        # save_dir = self.proj_path.joinpath('infer', 'whole')
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        if 'left' in self.config.infer_part:
            stem = torch.load(self.model_save_dir.joinpath('stem', 'stem_' + self.config.infer_name)).to(self.device)
        else:
            stem = torch.load(
                self.model_save_dir.joinpath(self.config.infer_part, 'stem_' + self.config.infer_name)
            ).to(self.device)
        D12 = torch.load(self.model_save_dir.joinpath(self.config.infer_part, 'D12_' + self.config.infer_name)).to(
            self.device
        )
        E2 = torch.load(self.model_save_dir.joinpath(self.config.infer_part, 'E2_' + self.config.infer_name)).to(
            self.device
        )
        D2 = torch.load(self.model_save_dir.joinpath(self.config.infer_part, 'D2_' + self.config.infer_name)).to(
            self.device
        )
        cols = ['name', 'acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']
        stem_seg_results = pd.DataFrame(columns=cols)  # 保存每张图的结果
        asm_results = pd.DataFrame(columns=cols)  # 保存每张图的结果
        init_results = pd.DataFrame(columns=cols)  # 保存每张图的结果
        dice_results = pd.DataFrame(columns=['name', 'seg', 'asm', 'asm-seg'])
        i = 0
        for image, label, all_pts_hm, shape, b, image_name in tqdm(data_loader):
            image_name = image_name[0]
            stem_seg_result = [image_name]
            asm_result = [image_name]
            init_result = [image_name]
            dice_result = [image_name]
            image = image.to(self.device)
            with torch.no_grad():
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                if self.config.dataset_name == 'ms':
                    initial_shape_L, initial_shape_hm_L = self.init_shape(
                        key_pts_hm_pred[:, :4],
                        key_pts_coord_pred[:, :4],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 0],
                        26,
                    )
                    initial_shape_R, initial_shape_hm_R = self.init_shape(
                        key_pts_hm_pred[:, 4:],
                        key_pts_coord_pred[:, 4:],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 1],
                        26,
                    )
                    initial_shape = np.concatenate((initial_shape_L, initial_shape_R), 1)
                    initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
                else:  # CAMUS
                    initial_shape, initial_shape_hm = self.init_shape(
                        key_pts_hm_pred,
                        key_pts_coord_pred,
                        self.config.pt_idxes,
                        half_seg_pred,
                        self.config.total_pts_num,
                    )
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
                if torch.isnan(all_pts_hm_pred).any():
                    print('eval NaN:', image_name)
                    continue
            # 保存all_pts_hm_pred as tensor
            if save_dir.parent.name == 'gen_unlabeled':
                torch.save(
                    all_pts_hm_pred.squeeze(0).detach().cpu(),
                    save_dir.joinpath(image_name[:-4] + '_all_pts_hm_pred.pt'),
                )
                continue

            # 得到seg，与label计算分割指标
            if self.config.dataset_name == 'ms':
                seg_pred_L_binary, stem_seg_hd95_L, stem_seg_assd_L = self.evaluate_hd_assd(seg_pred[:, 0], label[:, 0])
                seg_pred_R_binary, stem_seg_hd95_R, stem_seg_assd_R = self.evaluate_hd_assd(seg_pred[:, 1], label[:, 1])
                seg_pred_binary = seg_pred_L_binary + seg_pred_R_binary
                stem_seg_hd95 = (stem_seg_hd95_L + stem_seg_hd95_R) / 2
                stem_seg_assd = (stem_seg_assd_L + stem_seg_assd_R) / 2
            else:
                seg_pred_binary, stem_seg_hd95, stem_seg_assd = self.evaluate_hd_assd(seg_pred.squeeze(), label)
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg_binary.png')), seg_pred_binary)
            seg_pred_scores = all_score(torch.sum(seg_pred, 1).cpu().numpy(), torch.sum(label, 1).numpy())
            stem_seg_result = stem_seg_result + list(seg_pred_scores) + [stem_seg_hd95, stem_seg_assd]
            stem_seg_results.loc[i] = stem_seg_result
            # for stem vis
            mask = self.vis_key_pts(
                image.cpu().squeeze(),
                key_pts_coord_pred.squeeze() * self.config.resize_ratio,
                shape[:, self.config.pt_idxes].squeeze() * self.config.resize_ratio,
                self.config.pt_idxes,
            )
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_key.png')), mask)
            # seg_pred_img = transforms.ToPILImage()(torch.sum(seg_pred, 1).squeeze(0).cpu())
            # seg_pred_img.save(save_dir.joinpath(image_name[:-4] + '_seg.png'))
            # for j in range(key_pts_hm_pred.shape[1]):
            #     pt_img = key_pts_hm_pred[0, j]
            #     pt_img = transforms.ToPILImage()(pt_img)
            #     pt_idx = self.config.pt_idxes[j]
            #     pt_img.save(save_dir.joinpath('{}_{}.png'.format(image_name[:-4], pt_idx)))

            # 得到asm_shape，与label计算分割指标
            all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
            if self.config.dataset_name == 'ms':
                b_pred_L, asm_shape_L = self.sasm.transform(all_pts_coord_pred[0, :26])
                b_pred_R, asm_shape_R = self.sasm.transform(all_pts_coord_pred[0, 26:])
                b_pred = torch.cat((b_pred_L, b_pred_R), dim=0)
                asm_shape = torch.cat((asm_shape_L, asm_shape_R), dim=0)
                shape_mask_L_np = shape_to_mask(asm_shape_L.cpu(), self.config.input_size)
                shape_mask_L_np = cv2.resize(shape_mask_L_np, (label.shape[-1], label.shape[-2]))
                asm_L_hd95 = binary.hd95(shape_mask_L_np, label[:, 0].squeeze().numpy())
                asm_L_assd = binary.assd(shape_mask_L_np, label[:, 0].squeeze().numpy())
                shape_mask_R_np = shape_to_mask(asm_shape_R.cpu(), self.config.input_size)
                shape_mask_R_np = cv2.resize(shape_mask_R_np, (label.shape[-1], label.shape[-2]))
                asm_R_hd95 = binary.hd95(shape_mask_R_np, label[:, 1].squeeze().numpy())
                asm_R_assd = binary.assd(shape_mask_R_np, label[:, 1].squeeze().numpy())
                shape_mask_np = shape_mask_L_np + shape_mask_R_np
                asm_hd95 = (asm_L_hd95 + asm_R_hd95) / 2
                asm_assd = (asm_L_assd + asm_R_assd) / 2
            else:
                b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[0])
                shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
                shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
                asm_hd95 = binary.hd95(shape_mask_np, label.squeeze().numpy())
                asm_assd = binary.assd(shape_mask_np, label.squeeze().numpy())
            asm_scores = all_score(shape_mask_np, torch.sum(label, 1).numpy())
            asm_result = asm_result + list(asm_scores) + [asm_hd95, asm_assd]
            asm_results.loc[i] = asm_result
            dice_result = dice_result + [seg_pred_scores[-3], asm_scores[-3], asm_scores[-3] - seg_pred_scores[-3]]
            dice_results.loc[i] = dice_result

            # 计算init shape的指标
            if self.config.dataset_name == 'ms':
                init_shape_mask_L = shape_to_mask(initial_shape_L, self.config.input_size)
                init_shape_mask_L = cv2.resize(init_shape_mask_L, (label.shape[-1], label.shape[-2]))
                init_shape_hd95_L = binary.hd95(init_shape_mask_L, label[:, 0].squeeze().numpy())
                init_shape_assd_L = binary.assd(init_shape_mask_L, label[:, 0].squeeze().numpy())
                init_shape_mask_R = shape_to_mask(initial_shape_R, self.config.input_size)
                init_shape_mask_R = cv2.resize(init_shape_mask_R, (label.shape[-1], label.shape[-2]))
                init_shape_hd95_R = binary.hd95(init_shape_mask_R, label[:, 1].squeeze().numpy())
                init_shape_assd_R = binary.assd(init_shape_mask_R, label[:, 1].squeeze().numpy())
                init_shape_mask = init_shape_mask_L + init_shape_mask_R
                init_shape_hd95 = (init_shape_hd95_L + init_shape_hd95_R) / 2
                init_shape_assd = (init_shape_assd_L + init_shape_assd_R) / 2
            else:
                init_shape_mask = shape_to_mask(initial_shape, self.config.input_size)
                init_shape_mask = cv2.resize(init_shape_mask, (label.shape[-1], label.shape[-2]))
                init_shape_hd95 = binary.hd95(init_shape_mask, label.squeeze().numpy())
                init_shape_assd = binary.assd(init_shape_mask, label.squeeze().numpy())
            init_shape_scores = all_score(init_shape_mask, torch.sum(label, 1).squeeze().numpy())
            init_result = init_result + list(init_shape_scores) + [init_shape_hd95, init_shape_assd]
            init_results.loc[i] = init_result
            i += 1

            # for vis
            init_pts_vis = self.vis_all_pts(
                image.cpu().squeeze(), initial_shape * self.config.resize_ratio, False, label.squeeze().numpy()
            )
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_pts_init.png')), init_pts_vis)
            pts_vis = self.vis_all_pts(
                image.cpu().squeeze(),
                all_pts_coord_pred.detach().cpu()[0] * self.config.resize_ratio,
                False,
                label.squeeze().numpy(),
            )
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_pts_refine.png')), pts_vis)
            seg_vis = self.vis_pred_boundary(image.cpu().squeeze(), label.squeeze().numpy(), shape_mask_np, None)
            cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_asm.png')), seg_vis)
            # seg_pred = transforms.ToPILImage()(seg_pred.squeeze(0))
            # seg_pred.save(save_dir.joinpath(image_name[:-4] + '_seg.png'))
            # break
        # 保存数值结果
        if save_dir.parent.name != 'gen_unlabeled':
            mean_idx = i
            std_idx = i + 1
            mean = stem_seg_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].mean()
            result = ['mean'] + list(mean)
            stem_seg_results.loc[mean_idx] = result
            std = stem_seg_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].std()
            result = ['std'] + list(std)
            stem_seg_results.loc[std_idx] = result
            stem_seg_results.set_index(['name'], inplace=True)
            stem_seg_results.to_csv(save_dir.joinpath('stem_seg_results.csv'))

            mean = asm_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].mean()
            result = ['mean'] + list(mean)
            asm_results.loc[mean_idx] = result
            std = asm_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].std()
            result = ['std'] + list(std)
            asm_results.loc[std_idx] = result
            asm_results.set_index(['name'], inplace=True)
            asm_results.to_csv(save_dir.joinpath('asm_results.csv'))

            mean = init_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].mean()
            result = ['mean'] + list(mean)
            init_results.loc[mean_idx] = result
            std = init_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].std()
            result = ['std'] + list(std)
            init_results.loc[std_idx] = result
            init_results.set_index(['name'], inplace=True)
            init_results.to_csv(save_dir.joinpath('init_results.csv'))

            mean = dice_results[['acc', 'auc', 'sen', 'spe', 'pre', 'iou', 'dice', 'hd95', 'assd']].mean()
            result = ['mean'] + list(mean)
            dice_results.loc[mean_idx] = result
            std = dice_results[['seg', 'asm', 'asm-seg']].std()
            result = ['std'] + list(std)
            dice_results.loc[std_idx] = result
            dice_results.set_index(['name'], inplace=True)
            dice_results.to_csv(save_dir.joinpath('dice_results.csv'))

    def train_finetune(self):
        '''
        加载所有部分，微调
        '''
        data_loader = self.dataloader.load_data(self.config.train_mode, self.config.train_batch_size)
        vis_dir = self.vis_train_dir.joinpath(self.config.train_mode, 'finetune')
        model_save_dir = self.model_save_dir.joinpath('finetune')
        for d in [vis_dir, model_save_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        logger = config_log(vis_dir)
        minLoss = 1e12
        bestDice = 0.0
        stem = torch.load(self.model_save_dir.joinpath('stem', 'stem_minLoss')).to(self.device)
        D12 = torch.load(self.model_save_dir.joinpath('left', 'D12_bestDice')).to(self.device)
        E2 = torch.load(self.model_save_dir.joinpath('left', 'E2_bestDice')).to(self.device)
        D2 = torch.load(self.model_save_dir.joinpath('left', 'D2_bestDice')).to(self.device)
        stem.train()
        D12.train()
        E2.train()
        D2.train()
        optimizer = torch.optim.Adam(
            [
                {'params': stem.parameters()},
                {'params': D12.parameters()},
                {'params': E2.parameters()},
                {'params': D2.parameters()},
            ],
            lr=self.config.lr / 1e2,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        for epoch in tqdm(range(1, self.config.epochs + 1)):
            for image, label, all_pts_hm, shape, b, image_name in data_loader:
                image, label, all_pts_hm = image.to(self.device), label.to(self.device), all_pts_hm.to(self.device)
                shape, b = shape.to(self.device), b.to(self.device)
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                # initial_shape, initial_shape_hm = self.init_shape(key_pts_hm_pred, key_pts_coord_pred, half_seg_pred)
                if self.config.dataset_name == 'ms':
                    initial_shape_L, initial_shape_hm_L = self.init_shape(
                        key_pts_hm_pred[:, :4],
                        key_pts_coord_pred[:, :4],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 0],
                        26,
                    )
                    initial_shape_R, initial_shape_hm_R = self.init_shape(
                        key_pts_hm_pred[:, 4:],
                        key_pts_coord_pred[:, 4:],
                        self.config.pt_idxes[:4],
                        half_seg_pred[:, 1],
                        26,
                    )
                    initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
                else:  # CAMUS
                    initial_shape, initial_shape_hm = self.init_shape(
                        key_pts_hm_pred,
                        key_pts_coord_pred,
                        self.config.pt_idxes,
                        half_seg_pred,
                        self.config.total_pts_num,
                    )
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
                all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
                bs_pred = []
                for idx in range(image.shape[0]):
                    if self.config.dataset_name == 'ms':
                        b_pred_L, _ = self.sasm.transform(all_pts_coord_pred[idx, :26])
                        b_pred_R, _ = self.sasm.transform(all_pts_coord_pred[idx, 26:])
                        b_pred = torch.cat((b_pred_L, b_pred_R), dim=0)
                    else:
                        b_pred, _ = self.sasm.transform(all_pts_coord_pred[idx])  # [10]
                    bs_pred.append(b_pred)
                bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
                # calculate loss
                seg_loss = self.Loss(seg_pred, label)
                key_pts_hm_loss = self.Loss(key_pts_hm_pred, all_pts_hm[:, self.config.pt_idxes])
                # key_pts_coord_loss = self.Loss(key_pts_coord_pred, shape[:, self.config.pt_idxes])
                all_pts_hm_loss = self.Loss(all_pts_hm_pred, all_pts_hm)
                # all_pts_coord_loss = self.Loss(all_pts_coord_pred, shape)
                b_loss = self.Loss(bs_pred, b) / 1e2
                loss = seg_loss + key_pts_hm_loss + all_pts_hm_loss + b_loss
                # if epoch > self.config.epochs // 2:
                #     b_loss = self.Loss(bs_pred, b) / 1e2
                #     loss = loss + b_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break
            scheduler.step()
            if epoch % self.config.step == 0:
                evaluation, val_losses = self.evaluate_left([stem, D12, E2, D2], vis_dir)
                acc, auc, sen, spe, pre, iou, dc = evaluation
                logger.info(
                    '{}:\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'.format(
                        epoch, acc, auc, sen, spe, pre, iou, dc
                    )
                )
                logger.info(
                    '{}:\t {:.8f}\t {:.8f}\t {:.4f}\t {:.8f}\t {:.4f}, {:.4f}'.format(
                        epoch, val_losses[0], val_losses[1], val_losses[2], val_losses[3], val_losses[4], val_losses[5]
                    )
                )
                if sum(val_losses) < minLoss:
                    minLoss = sum(val_losses)
                    torch.save(stem, model_save_dir.joinpath('stem_minLoss'))
                    torch.save(D12, model_save_dir.joinpath('D12_minLoss'))
                    torch.save(E2, model_save_dir.joinpath('E2_minLoss'))
                    torch.save(D2, model_save_dir.joinpath('D2_minLoss'))
                    logger.info('Saving minLoss......')
                if dc > bestDice:
                    bestDice = dc
                    torch.save(stem, model_save_dir.joinpath('stem_bestDice'))
                    torch.save(D12, model_save_dir.joinpath('D12_bestDice'))
                    torch.save(E2, model_save_dir.joinpath('E2_bestDice'))
                    torch.save(D2, model_save_dir.joinpath('D2_bestDice'))
                    logger.info('Saving bestDice......')
        # 保存一个最终模型
        torch.save(stem, model_save_dir.joinpath('stem_final'))
        torch.save(D12, model_save_dir.joinpath('D12_final'))
        torch.save(E2, model_save_dir.joinpath('E2_final'))
        torch.save(D2, model_save_dir.joinpath('D2_final'))

    def train_whole(self):
        '''
        加载所有部分，微调
        '''
        dataloader = self.dataloader.load_data(self.config.train_mode, self.config.train_batch_size)
        vis_dir = self.vis_train_dir.joinpath(self.config.train_mode, 'whole')
        if not vis_dir.exists():
            vis_dir.mkdir(parents=True)
        logger = config_log(vis_dir)
        bestDice = 0.0
        stem = self.config.stem(1, self.config.channels, self.config.stem_out_ch).to(self.device)
        D12 = self.config.d4_net(self.config.channels, False, 1, self.config.decoder_conv).to(self.device)
        E2 = self.config.e4_net(
            self.config.channels[1] + self.config.total_pts_num + self.config.stem_out_ch[0],
            self.config.channels,
            self.config.decoder_conv,
        ).to(self.device)
        D2 = self.config.d4_net(self.config.channels, True, self.config.total_pts_num, self.config.decoder_conv).to(
            self.device
        )
        stem.train()
        D12.train()
        E2.train()
        D2.train()
        optimizer = torch.optim.Adam(
            [
                {'params': stem.parameters()},
                {'params': D12.parameters()},
                {'params': E2.parameters()},
                {'params': D2.parameters()},
            ],
            lr=self.config.lr / 1e2,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        for epoch in tqdm(range(1, self.config.epochs + 1)):
            for image, label, all_pts_hm, shape, b, image_name in dataloader:
                image, label, all_pts_hm = image.to(self.device), label.to(self.device), all_pts_hm.to(self.device)
                shape, b = shape.to(self.device), b.to(self.device)
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                initial_shape, initial_shape_hm = self.init_shape(key_pts_hm_pred, key_pts_coord_pred, half_seg_pred)
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
                all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
                bs_pred = []
                for idx in range(image.shape[0]):
                    b_pred, _ = self.sasm.transform(all_pts_coord_pred[idx])  # [10]
                    bs_pred.append(b_pred)
                bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
                # calculate loss
                seg_loss = self.Loss(seg_pred, label)
                key_pts_hm_loss = self.Loss(key_pts_hm_pred, all_pts_hm[:, self.config.pt_idxes])
                all_pts_hm_loss = self.Loss(all_pts_hm_pred, all_pts_hm)
                loss = seg_loss + key_pts_hm_loss + all_pts_hm_loss
                if epoch > 200:
                    b_loss = self.Loss(bs_pred, b) / 1e2
                    loss = loss + b_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # break
            scheduler.step()
            if epoch % self.config.step == 0:
                evaluation, val_losses = self.evaluate_whole([stem, D12, E2, D2], vis_dir)
                acc, auc, sen, spe, pre, iou, dc = evaluation
                logger.info(
                    '{}:\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'.format(
                        epoch, acc, auc, sen, spe, pre, iou, dc
                    )
                )
                logger.info(
                    '{}:\t {:.8f}\t {:.8f}\t {:.4f}\t {:.8f}\t {:.4f}, {:.4f}'.format(
                        epoch, val_losses[0], val_losses[1], val_losses[2], val_losses[3], val_losses[4], val_losses[5]
                    )
                )
                if dc > bestDice:
                    bestDice = dc
                    torch.save(stem, self.model_save_dir.joinpath('stem_FbestDice'))
                    torch.save(D12, self.model_save_dir.joinpath('D12_FbestDice'))
                    torch.save(E2, self.model_save_dir.joinpath('E2_FbestDice'))
                    torch.save(D2, self.model_save_dir.joinpath('D2_FbestDice'))
                    logger.info('Saving......')
        # 保存一个最终模型
        torch.save(stem, self.model_save_dir.joinpath('stem_Ffinal'))
        torch.save(D12, self.model_save_dir.joinpath('D12_Ffinal'))
        torch.save(E2, self.model_save_dir.joinpath('E2_Ffinal'))
        torch.save(D2, self.model_save_dir.joinpath('D2_Ffinal'))

    def evaluate_whole(self, model_lst, vis_dir=None):
        stem, D12, E2, D2 = model_lst
        stem.eval()
        D12.eval()
        E2.eval()
        D2.eval()
        seg_loss = 0
        key_pts_hm_loss = 0
        key_pts_coord_loss = 0
        all_pts_hm_loss = 0.0
        all_pts_coord_loss = 0.0
        b_loss = 0.0
        evaluation = np.array([0.0 for _ in range(7)])
        with torch.no_grad():
            for image, label, all_pts_hm, shape, b, image_name in self.test_loader:
                image_name = image_name[0]
                image = image.to(self.device)
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                initial_shape, initial_shape_hm = self.init_shape(key_pts_hm_pred, key_pts_coord_pred, half_seg_pred)
                context = D12(emb1)
                # emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                emb2 = E2(torch.cat((initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
                all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
                bs_pred = []
                for idx in range(image.shape[0]):
                    b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[idx])  # [10]
                    bs_pred.append(b_pred)
                    # 得到shape，与label计算分割指标
                    shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
                    shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
                    temp = all_score(shape_mask_np, label.squeeze().numpy())
                    evaluation += temp
                    if vis_dir:
                        image = image.cpu().squeeze()
                        mask = self.vis_key_pts(
                            image,
                            key_pts_coord_pred.squeeze() * self.config.resize_ratio,
                            shape[:, self.config.pt_idxes].squeeze() * self.config.resize_ratio,
                            self.config.pt_idxes,
                        )
                        cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_key.png')), mask)
                        pts_vis = self.vis_all_pts(
                            image, all_pts_coord_pred.detach().cpu()[idx] * self.config.resize_ratio
                        )
                        cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_pts.png')), pts_vis)
                        seg_vis = self.vis_pred_boundary(image, label.squeeze().numpy(), shape_mask_np, temp[-1])
                        cv2.imwrite(str(vis_dir.joinpath(image_name[:-4] + '_asm.png')), seg_vis)
                        seg_pred_ = transforms.ToPILImage()(seg_pred.squeeze(0).cpu())
                        seg_pred_.save(vis_dir.joinpath(image_name[:-4] + '_seg.png'))
                bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
                # calculate loss
                seg_loss += self.Loss(seg_pred.detach().cpu(), label)
                key_pts_hm_loss = self.Loss(key_pts_hm_pred.detach().cpu(), all_pts_hm[:, self.config.pt_idxes])
                key_pts_coord_loss = self.Loss(key_pts_coord_pred.detach().cpu(), shape[:, self.config.pt_idxes])
                all_pts_hm_loss = self.Loss(all_pts_hm_pred.detach().cpu(), all_pts_hm)
                all_pts_coord_loss = self.Loss(all_pts_coord_pred.detach().cpu(), shape)
                b_loss = self.Loss(bs_pred.detach().cpu(), b)
                # break
        stem.train()
        D12.train()
        E2.train()
        D2.train()
        num = len(self.test_loader)
        losses = [
            seg_loss / num,
            key_pts_hm_loss / num,
            key_pts_coord_loss / num,
            all_pts_hm_loss / num,
            all_pts_coord_loss / num,
            b_loss / num,
        ]
        return evaluation / (num * self.config.val_batch_size), losses

    def infer_single_image_CAMUS(self):
        '''
        在训练完left后，infer整个流程
        '''
        # dataset
        root = Path(self.config.dataset_path)
        img_dir = root.joinpath('image_544x736')
        label_dir = root.joinpath('label_544x736')
        heatmap_dir = root.joinpath('heatmap', self.config.heatmap_dir_name)
        model_dir = self.model_save_dir.joinpath(self.config.infer_part)
        shapes = json.load(open(root.joinpath('shape_544x736.json')))
        # bs = json.load(open(Path(self.config.asm_save_dir).joinpath('b.json')))
        # load data
        image_names = [
            'patient0343_2CH_ES.png',
            'patient0318_2CH_ES.png',
            'patient0341_4CH_ED.png',
            'patient0317_4CH_ED.png',
        ]
        for image_name in image_names:
            img_path = img_dir.joinpath(image_name)
            image = Image.open(img_path)
            image = transforms.ToTensor()(image).unsqueeze(0)
            label_path = label_dir.joinpath(img_path.name)
            label = Image.open(label_path)
            label = transforms.ToTensor()(label).unsqueeze(0)
            label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))
            shape = torch.Tensor(shapes[image_name[:-4]]).unsqueeze(0)  # / torch.Tensor(self.config.input_size)
            shape = torch.div(shape, self.config.resize_ratio, rounding_mode='trunc')
            # b_gt = torch.Tensor(bs[image_name[:-4]]).unsqueeze(0)

            save_dir = self.proj_path.joinpath('infer', 'single_image')
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            stem = torch.load(model_dir.joinpath('stem_bestDice')).to(self.device)
            D12 = torch.load(model_dir.joinpath('D12_bestDice')).to(self.device)
            E2 = torch.load(model_dir.joinpath('E2_bestDice')).to(self.device)
            D2 = torch.load(model_dir.joinpath('D2_bestDice')).to(self.device)
            image = image.to(self.device)
            with torch.no_grad():
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                initial_shape, initial_shape_hm = self.init_shape(
                    key_pts_hm_pred, key_pts_coord_pred, self.config.pt_idxes, half_seg_pred, self.config.total_pts_num
                )
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
            all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
            b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[0])
            shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
            shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
            cv2.imwrite(str(save_dir.joinpath(image_name)), shape_mask_np * 255)

    def infer_single_image_CAMUS_MIA(self):
        '''
        在训练完left后，infer整个流程。取all_shape的pred_hms
        '''
        # dataset
        root = Path(self.config.dataset_path)
        img_dir = root.joinpath('image_544x736')
        label_dir = root.joinpath('label_544x736')
        heatmap_dir = root.joinpath('heatmap', self.config.heatmap_dir_name)
        model_dir = self.model_save_dir.joinpath(self.config.infer_part)
        shapes = json.load(open(root.joinpath('shape_544x736.json')))
        # bs = json.load(open(Path(self.config.asm_save_dir).joinpath('b.json')))
        # load data
        image_name = 'patient0348_2CH_ED.png'
        img_path = img_dir.joinpath(image_name)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image).unsqueeze(0)
        label_path = label_dir.joinpath(img_path.name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label).unsqueeze(0)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        save_dir = self.proj_path.joinpath('infer', 'single_image')
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = torch.load(model_dir.joinpath('stem_bestDice')).to(self.device)
        D12 = torch.load(model_dir.joinpath('D12_bestDice')).to(self.device)
        E2 = torch.load(model_dir.joinpath('E2_bestDice')).to(self.device)
        D2 = torch.load(model_dir.joinpath('D2_bestDice')).to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            seg_pred, key_pts_hm_pred, emb1 = stem(image)
            half_seg_pred = torch.nn.functional.interpolate(
                seg_pred,
                scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                mode='bilinear',
                align_corners=False,
            )
            key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
            initial_shape, initial_shape_hm, _ = self.init_shape(
                key_pts_hm_pred, key_pts_coord_pred, self.config.pt_idxes, half_seg_pred, self.config.total_pts_num
            )
            context = D12(emb1)
            emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
            all_pts_hm_pred = D2(emb2)
        torch.save(all_pts_hm_pred, save_dir.joinpath(image_name[:-4] + '_all_pts_hm_pred'))
        all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
        b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[0])
        shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
        shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
        cv2.imwrite(str(save_dir.joinpath(image_name)), shape_mask_np * 255)

    def infer_single_image_ms(self):
        '''
        在训练完left后，infer整个流程
        '''
        # dataset
        root = Path(self.config.dataset_path)
        img_dir = root.joinpath('image_960x832')
        label_dir = root.joinpath('label_single_960x832')
        # heatmap_dir = root.joinpath('heatmap', self.config.heatmap_dir_name)
        model_dir = self.model_save_dir.joinpath(self.config.infer_part)
        # shapes = json.load(open(root.joinpath('shape_544x736.json')))
        # bs = json.load(open(Path(self.config.asm_save_dir).joinpath('b.json')))
        # load data
        image_names = ['qg001-R2>2_G.png', 'QG22052901-R>162_G.png', 'QG091602-L>2_G.png']
        for image_name in image_names:
            img_path = img_dir.joinpath(image_name)
            image = Image.open(img_path)
            image = transforms.ToTensor()(image).unsqueeze(0)
            label_L_path = label_dir.joinpath(img_path.stem + '>L.png')
            label_L = Image.open(label_L_path)
            label_L = transforms.ToTensor()(label_L)
            label_L = torch.where(label_L > 0, torch.tensor(1.0), torch.tensor(0.0))
            label_R_path = label_dir.joinpath(img_path.stem + '>R.png')
            label_R = Image.open(label_R_path)
            label_R = transforms.ToTensor()(label_R)
            label_R = torch.where(label_R > 0, torch.tensor(1.0), torch.tensor(0.0))
            label = torch.cat([label_L, label_R], dim=0)
            # shape = torch.Tensor(shapes[image_name[:-4]]).unsqueeze(0)  # / torch.Tensor(self.config.input_size)
            # shape = torch.div(shape, self.config.resize_ratio, rounding_mode='trunc')
            # b_gt = torch.Tensor(bs[image_name[:-4]]).unsqueeze(0)

            save_dir = self.proj_path.joinpath('infer', 'single_image')
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            stem = torch.load(model_dir.joinpath('stem_bestDice')).to(self.device)
            D12 = torch.load(model_dir.joinpath('D12_bestDice')).to(self.device)
            E2 = torch.load(model_dir.joinpath('E2_bestDice')).to(self.device)
            D2 = torch.load(model_dir.joinpath('D2_bestDice')).to(self.device)
            image = image.to(self.device)
            with torch.no_grad():
                seg_pred, key_pts_hm_pred, emb1 = stem(image)
                half_seg_pred = torch.nn.functional.interpolate(
                    seg_pred,
                    scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                    mode='bilinear',
                    align_corners=False,
                )
                key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
                initial_shape_L, initial_shape_hm_L = self.init_shape(
                    key_pts_hm_pred[:, :4], key_pts_coord_pred[:, :4], self.config.pt_idxes[:4], half_seg_pred[:, 0], 26
                )
                initial_shape_R, initial_shape_hm_R = self.init_shape(
                    key_pts_hm_pred[:, 4:], key_pts_coord_pred[:, 4:], self.config.pt_idxes[:4], half_seg_pred[:, 1], 26
                )
                initial_shape = np.concatenate((initial_shape_L, initial_shape_R), 1)
                initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
                context = D12(emb1)
                emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
                all_pts_hm_pred = D2(emb2)
            all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
            b_pred_L, asm_shape_L = self.sasm.transform(all_pts_coord_pred[0, :26])
            b_pred_R, asm_shape_R = self.sasm.transform(all_pts_coord_pred[0, 26:])
            shape_mask_L_np = shape_to_mask(asm_shape_L.cpu(), self.config.input_size)
            shape_mask_L_np = cv2.resize(shape_mask_L_np, (label.shape[-1], label.shape[-2]))
            shape_mask_R_np = shape_to_mask(asm_shape_R.cpu(), self.config.input_size)
            shape_mask_R_np = cv2.resize(shape_mask_R_np, (label.shape[-1], label.shape[-2]))
            shape_mask_np = shape_mask_L_np + shape_mask_R_np
            cv2.imwrite(str(save_dir.joinpath(image_name)), shape_mask_np * 255)

        # bs_pred = []
        # b_pred = self.asm.reflex_shape(all_pts_coord_pred[0])  # [10]
        # bs_pred.append(b_pred)
        # bs_pred = torch.stack(bs_pred, 0)  # [B, 10]
        # # 得到seg，与label计算分割指标
        # seg_pred = seg_pred.detach().cpu()
        # # seg_temp = all_score(seg_pred.numpy(), label.numpy())
        # # 得到asm_shape，与label计算分割指标
        # asm_shape = self.asm.validate_shape(all_pts_coord_pred.detach()[0])
        # shape_mask_np = shape_to_mask(asm_shape.cpu(), self.config.input_size)
        # shape_mask_np = cv2.resize(shape_mask_np, (label.shape[-1], label.shape[-2]))
        # # asm_temp = all_score(shape_mask_np, label.squeeze().numpy())
        # i += 1
        # # for vis
        # threshold = filters.threshold_otsu(seg_pred, nbins=256)
        # seg_pred_binary = torch.where(seg_pred > threshold, torch.tensor(1.0), torch.tensor(0.0))
        # seg_pred_binary = transforms.ToPILImage()(seg_pred_binary.squeeze(0))
        # seg_pred_binary.save(save_dir.joinpath(image_name[:-4] + '_seg_binary.png'))
        # seg_pred = transforms.ToPILImage()(seg_pred.squeeze(0))
        # seg_pred.save(save_dir.joinpath(image_name[:-4] + '_seg.png'))
        # init_shape_coord = self.init_shape_coord(key_pts_coord_pred, seg_binary)
        # init_pts_vis = self.vis_all_pts(image.cpu().squeeze(), init_shape_coord.squeeze() * self.config.resize_ratio, False)
        # cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_pts_init.png')), init_pts_vis)
        # pts_vis = self.vis_all_pts(image.cpu().squeeze(), all_pts_coord_pred.detach().cpu()[0] * self.config.resize_ratio, False)
        # cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_pts_refine.png')), pts_vis)
        # # original UNet seg
        # # seg_pred_path = '/4t/work/ASM/output/CAMUS/resunet/2/output/seg_binary/' + image_name
        # seg_pred_path = '/4t/work/ASM/output/CM540/unet/1/output/0.9042/output/seg_binary/' + image_name
        # seg_binary = cv2.imread(seg_pred_path, -1)
        # seg_vis = self.vis_pred_boundary_and_asm(image.cpu().squeeze(), label.squeeze().numpy(), seg_binary, shape_mask_np, None)
        # cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_asm.png')), seg_vis)

    def infer_single_image_ablation_CAMUS(self):
        '''
        画中间过程图，黑白图，label的轮廓当背景，展示关键点的演化。消融实验画图才需要
        '''
        # dataset
        root = Path(self.config.dataset_path)
        img_dir = root.joinpath('image_544x736')
        label_dir = root.joinpath('label_544x736')
        heatmap_dir = root.joinpath('heatmap', self.config.heatmap_dir_name)
        model_dir = self.model_save_dir.joinpath(self.config.infer_part)
        # load data
        # image_name = 'patient0319_4CH_ED.png'
        image_name = 'patient0341_4CH_ED.png'
        img_path = img_dir.joinpath(image_name)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image).unsqueeze(0)
        label_path = label_dir.joinpath(img_path.name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label).unsqueeze(0)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        save_dir = self.proj_path.joinpath('infer', 'single_image', image_name[:-4])
        print(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        stem = torch.load(model_dir.joinpath('stem_bestDice')).to(self.device)
        D12 = torch.load(model_dir.joinpath('D12_bestDice')).to(self.device)
        E2 = torch.load(model_dir.joinpath('E2_bestDice')).to(self.device)
        D2 = torch.load(model_dir.joinpath('D2_bestDice')).to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            seg_pred, key_pts_hm_pred, emb1 = stem(image)
            half_seg_pred = torch.nn.functional.interpolate(
                seg_pred,
                scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                mode='bilinear',
                align_corners=False,
            )
            key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
            initial_shape, initial_shape_hm, _ = self.init_shape(
                key_pts_hm_pred, key_pts_coord_pred, self.config.pt_idxes, half_seg_pred, self.config.total_pts_num
            )
            context = D12(emb1)
            emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
            all_pts_hm_pred = D2(emb2)
        all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
        seg_pred = seg_pred.detach().cpu()
        b_pred, asm_shape = self.sasm.transform(all_pts_coord_pred[0])

        # for vis
        label = label.squeeze().numpy()
        label = label.astype(np.uint8) * 255  # .transpose(1, 2, 0)
        mask = np.zeros((label.shape[0], label.shape[1], 3))
        edge = cv2.Canny(label, 30, 100)
        contour = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        seg_pred = seg_pred.squeeze().numpy()
        seg_pred = cv2.cvtColor(seg_pred * 255, cv2.COLOR_GRAY2BGR).astype(np.float64)
        cv2.drawContours(seg_pred, contour, -1, (0, 255, 0), 2)  # 绿色
        base_mask = mask + seg_pred
        # base_mask = cv2.addWeighted(mask, 1.2, seg_pred, 1, 0)

        # fig1: label+seg+KLs
        from copy import deepcopy

        mask = deepcopy(base_mask)
        # mask_KLs = self.vis_key_pts(mask, key_pts_coord_pred.squeeze() * self.config.resize_ratio, False)
        key_pts_coord_pred = key_pts_coord_pred * self.config.resize_ratio
        for i in range(key_pts_coord_pred.shape[1]):
            cv2.circle(mask, (int(key_pts_coord_pred[0][i][0]), int(key_pts_coord_pred[0][i][1])), 2, (0, 0, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+KLs.png')), mask)

        # fig2: based fig1, 增加CLs
        # init_shape_coord = self.init_shape_coord(key_pts_coord_pred)
        # initial_shape, initial_shape_hm = self.init_shape(
        #     key_pts_hm_pred, key_pts_coord_pred, self.config.pt_idxes, half_seg_pred, self.config.total_pts_num
        # )
        for i in range(initial_shape.squeeze().shape[0]):
            if i in self.config.pt_idxes:
                continue
            cv2.circle(mask, (int(initial_shape[i][0]), int(initial_shape[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+KLs+CLs.png')), mask)

        # fig3: label+seg+refined shape
        mask = deepcopy(base_mask)
        all_pts_coord_pred = all_pts_coord_pred.detach().cpu().squeeze().numpy() * self.config.resize_ratio
        for i in range(all_pts_coord_pred.shape[0]):
            if i in self.config.pt_idxes:
                cv2.circle(mask, (int(all_pts_coord_pred[i][0]), int(all_pts_coord_pred[i][1])), 2, (0, 0, 255), 5)
            else:
                cv2.circle(mask, (int(all_pts_coord_pred[i][0]), int(all_pts_coord_pred[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+refined.png')), mask)

        # fig4: final output
        mask = deepcopy(base_mask)
        asm_shape = asm_shape.cpu() * self.config.resize_ratio
        for i in range(asm_shape.shape[0]):
            if i in self.config.pt_idxes:
                cv2.circle(mask, (int(asm_shape[i][0]), int(asm_shape[i][1])), 2, (0, 0, 255), 5)
            else:
                cv2.circle(mask, (int(asm_shape[i][0]), int(asm_shape[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+final.png')), mask)

    def infer_single_image_ablation_ms(self):
        '''
        画中间过程图，黑白图，label的轮廓当背景，展示关键点的演化。消融实验画图才需要
        '''
        # dataset
        root = Path(self.config.dataset_path)
        img_dir = root.joinpath('image_960x832')
        label_dir = root.joinpath('label_single_960x832')
        heatmap_dir = root.joinpath('heatmap', self.config.heatmap_dir_name)
        model_dir = self.model_save_dir.joinpath(self.config.infer_part)
        # load data
        # image_name = 'gq0811-L>2_G.png'
        image_name = 'qg001-R2>2_G.png'
        img_path = img_dir.joinpath(image_name)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image).unsqueeze(0)
        label_L_path = label_dir.joinpath(img_path.stem + '>L.png')
        label_L = Image.open(label_L_path)
        label_L = transforms.ToTensor()(label_L)
        label_L = torch.where(label_L > 0, torch.tensor(1.0), torch.tensor(0.0))
        label_R_path = label_dir.joinpath(img_path.stem + '>R.png')
        label_R = Image.open(label_R_path)
        label_R = transforms.ToTensor()(label_R)
        label_R = torch.where(label_R > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = torch.cat([label_L, label_R], dim=0)

        save_dir = self.proj_path.joinpath('infer', 'single_image', image_name[:-4])
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        stem = torch.load(model_dir.joinpath('stem_bestDice')).to(self.device)
        D12 = torch.load(model_dir.joinpath('D12_bestDice')).to(self.device)
        E2 = torch.load(model_dir.joinpath('E2_bestDice')).to(self.device)
        D2 = torch.load(model_dir.joinpath('D2_bestDice')).to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            seg_pred, key_pts_hm_pred, emb1 = stem(image)
            half_seg_pred = torch.nn.functional.interpolate(
                seg_pred,
                scale_factor=(1 / self.config.resize_ratio, 1 / self.config.resize_ratio),
                mode='bilinear',
                align_corners=False,
            )
            key_pts_coord_pred = soft_argmax_tensor(key_pts_hm_pred)
            initial_shape_L, initial_shape_hm_L, shapes_L = self.init_shape(
                key_pts_hm_pred[:, :4], key_pts_coord_pred[:, :4], self.config.pt_idxes[:4], half_seg_pred[:, 0], 26
            )
            initial_shape_R, initial_shape_hm_R, shapes_R = self.init_shape(
                key_pts_hm_pred[:, 4:], key_pts_coord_pred[:, 4:], self.config.pt_idxes[:4], half_seg_pred[:, 1], 26
            )
            initial_shape = np.concatenate((initial_shape_L, initial_shape_R), 0)
            initial_shape_hm = torch.cat((initial_shape_hm_L, initial_shape_hm_R), dim=1)
            context = D12(emb1)
            emb2 = E2(torch.cat((half_seg_pred, initial_shape_hm, context), dim=1))
            all_pts_hm_pred = D2(emb2)
        all_pts_coord_pred = soft_argmax_tensor(all_pts_hm_pred)  # [B, N, 2]
        seg_pred = seg_pred.detach().cpu().squeeze()
        b_pred_L, asm_shape_L = self.sasm.transform(all_pts_coord_pred[0, :26])
        b_pred_R, asm_shape_R = self.sasm.transform(all_pts_coord_pred[0, 26:])
        asm_shape = torch.cat((asm_shape_L, asm_shape_R), dim=0)

        # for vis
        label_np = label.squeeze().numpy()
        label_np = label_np.astype(np.uint8) * 255  # .transpose(1, 2, 0)
        # mask = np.zeros((label_np.shape[-2], label_np.shape[-1], 3))
        edge_L = cv2.Canny(label_np[0], 30, 100)
        edge_R = cv2.Canny(label_np[1], 30, 100)
        contour_L = cv2.findContours(edge_L, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour_R = cv2.findContours(edge_R, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # seg_pred二值化
        seg_pred_L_binary, stem_seg_hd95_L, stem_seg_assd_L = self.evaluate_hd_assd(seg_pred[0], label[0])
        seg_pred_R_binary, stem_seg_hd95_R, stem_seg_assd_R = self.evaluate_hd_assd(seg_pred[1], label[1])
        seg_pred_binary = seg_pred_L_binary + seg_pred_R_binary
        # seg_pred_binary = np.concatenate((seg_pred_L_binary, seg_pred_R_binary), 0)
        # seg_pred_binary = np.argmax(seg_pred_binary, axis=0)
        seg_pred_binary = seg_pred_binary.squeeze().astype(np.uint8)
        seg_pred_binary = cv2.cvtColor(seg_pred_binary, cv2.COLOR_GRAY2BGR).astype(np.float64)
        cv2.drawContours(seg_pred_binary, contour_L, -1, (0, 255, 0), 2)  # 绿色
        cv2.drawContours(seg_pred_binary, contour_R, -1, (0, 255, 0), 2)  # 绿色
        base_mask = seg_pred_binary

        # fig1: label+seg+KLs
        from copy import deepcopy

        mask = deepcopy(base_mask)
        # mask_KLs = self.vis_key_pts(mask, key_pts_coord_pred.squeeze() * self.config.resize_ratio, False)
        key_pts_coord_pred = key_pts_coord_pred * self.config.resize_ratio
        for i in range(key_pts_coord_pred.shape[1]):
            cv2.circle(mask, (int(key_pts_coord_pred[0][i][0]), int(key_pts_coord_pred[0][i][1])), 2, (0, 0, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+KLs.png')), mask)

        # fig2: based fig1, 增加CLs
        initial_shape_L_ = shapes_L[2]
        initial_shape_R_ = shapes_R[6]
        initial_shape = np.concatenate((initial_shape_L_, initial_shape_R_), 0)
        initial_shape = initial_shape * self.config.resize_ratio
        for i in range(initial_shape.squeeze().shape[0]):
            if i in self.config.pt_idxes:
                continue
            cv2.circle(mask, (int(initial_shape[i][0]), int(initial_shape[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+KLs+CLs.png')), mask)

        # fig3: label+seg+refined shape
        mask = deepcopy(base_mask)
        all_pts_coord_pred = all_pts_coord_pred.detach().cpu().squeeze().numpy() * self.config.resize_ratio
        for i in range(all_pts_coord_pred.shape[0]):
            if i in self.config.pt_idxes:
                cv2.circle(mask, (int(all_pts_coord_pred[i][0]), int(all_pts_coord_pred[i][1])), 2, (0, 0, 255), 5)
            else:
                cv2.circle(mask, (int(all_pts_coord_pred[i][0]), int(all_pts_coord_pred[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+refined.png')), mask)

        # fig4: final output
        mask = deepcopy(base_mask)
        asm_shape = asm_shape.cpu() * self.config.resize_ratio
        for i in range(asm_shape.shape[0]):
            if i in self.config.pt_idxes:
                cv2.circle(mask, (int(asm_shape[i][0]), int(asm_shape[i][1])), 2, (0, 0, 255), 5)
            else:
                cv2.circle(mask, (int(asm_shape[i][0]), int(asm_shape[i][1])), 2, (0, 140, 255), 5)
        cv2.imwrite(str(save_dir.joinpath(image_name[:-4] + '_seg+final.png')), mask)


def main(config):
    trainer = Trainer_DSM(config)

    # Step1: train labeled
    # trainer.train_stem()
    # trainer.train_left()
    # trainer.train_finetune()

    # Step2: infer testset for the first time
    # save_dir = trainer.proj_path.joinpath('infer', 'whole')
    # trainer.infer_whole(trainer.test_loader, save_dir)

    # Step3: generate unlabeled psuedo label
    # gen_unlabeled_loader = trainer.dataloader.load_data('gen_unlabeled', config.val_batch_size)
    # save_dir = config.save_path.joinpath('gen_unlabeled', 'heatmap_tensor')
    # trainer.infer_whole(gen_unlabeled_loader, save_dir)

    # if config.infer_part == 'stem':
    #     trainer.infer_stem()
    # else:
    #     trainer.infer_whole(trainer.test_loader, trainer.proj_path.joinpath('infer', config.train_mode, config.infer_part, config.infer_name))

    if config.dataset_name == 'CAMUS':
        # trainer.infer_single_image_CAMUS()
        # trainer.infer_single_image_ablation_CAMUS()
        trainer.infer_single_image_CAMUS_MIA()
    elif config.dataset_name == 'ms':
        # trainer.infer_single_image_ms()
        trainer.infer_single_image_ablation_ms()
    else:
        pass
