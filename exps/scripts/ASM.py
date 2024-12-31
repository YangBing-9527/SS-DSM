import math
import cv2
import json
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from models.ASM import ActiveShapeModel
from models.utils import shape_to_iou, vis_pts_polylines


class ASM_process:
    '''
    调用ASM，实现多种功能
    '''

    def __init__(self, config) -> None:
        self.config = config
        self.dataset_path = Path(self.config.dataset_path)
        self.dataset_pts = json.load(open(self.dataset_path.joinpath('shape_544x736.json')))
        self.proj_dir = Path(self.config.asm_dir)
        if not self.proj_dir.exists():
            self.proj_dir.mkdir(parents=True)
        # self.imgWH = [self.config.input_size[0] * self.config.resize_ratio, self.config.input_size[1] * self.config.resize_ratio]
        self.imgWH = [x * self.config.resize_ratio for x in self.config.input_size]

    def fit_transform_labeled(self, data_weights=None):
        '''
        在labeled data上建模，transform labeled data 和 testset
        data_weights: [N, 51, 2]，记录每个点的权重，labeled全是1，infered是小数
        vis: bool，是否可视化transform后的结果
        '''
        import warnings

        warnings.filterwarnings('ignore')
        # Part I: fit
        dataset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', str(self.config.labeled_ratio) + '.json')))
        trainset_pts = {k: self.dataset_pts[k[:-4]] for k in dataset_names['labeled']}
        trainset_pts_tensor = torch.Tensor([v for v in trainset_pts.values()])  # [544, 51, 2]
        model = ActiveShapeModel(trainset_pts_tensor, data_weights)
        model.save(self.proj_dir.joinpath('model'))

        # Part II: transform
        img_dir = self.dataset_path.joinpath('image_544x736')
        label_dir = self.dataset_path.joinpath('label_544x736')
        trainset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', str(self.config.labeled_ratio) + '.json')))
        testset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', 'test.json')))['test']
        labeled_names = trainset_names['labeled']
        d = {'labeled': labeled_names, 'testset': testset_names}
        for key, names in d.items():
            vis_dir = self.proj_dir.joinpath('check_label', key)
            if not vis_dir.exists():
                vis_dir.mkdir(parents=True)
            ious = 0
            b_resuls = {}
            iou_resuls = {}
            for img_name in names:
                img_path = img_dir.joinpath(img_name)
                image = cv2.imread(str(img_path))
                image = image / np.max(image)  # [H, W]
                label_path = label_dir.joinpath(img_name)
                label = cv2.imread(str(label_path), -1)
                pts = torch.Tensor(self.dataset_pts[img_path.stem])  # [51, 2], HW
                flip = False
                if pts[0, 0] < pts[20, 0]:  # 如果平坦部朝左，则沿竖向中轴线翻转坐标
                    pts[:, 0] = self.config.input_size[0] - pts[:, 0]
                    flip = True
                b, shape = model.transform(pts)
                if flip:
                    shape[:, 0] = self.config.input_size[0] - pts[:, 0]
                iou = shape_to_iou(shape, label // 255)
                vis_pts_polylines(
                    pts=shape,
                    WH=self.imgWH,
                    image=image,
                    label=label,
                    title='iou: {:.4f}'.format(iou),
                    save_path=vis_dir.joinpath(img_path.name),
                )
                iou_resuls[img_path.stem] = iou
                b_resuls[img_path.stem] = b.tolist()
                ious += iou
                # break
            iou_resuls['avg_iou'] = ious / len(names)
            iou_resuls = pd.DataFrame.from_dict(iou_resuls, orient='index')
            iou_resuls.to_csv(vis_dir.joinpath('iou.csv'))  # _no_cons
            b_resuls_pd = pd.DataFrame.from_dict(b_resuls, orient='index')
            b_resuls_pd.to_csv(self.proj_dir.joinpath('{}_b.csv'.format(key)))
            for index in range(len(b_resuls_pd.columns)):
                ax = b_resuls_pd[index]
                fig = sns.displot(ax, kde=True)
                fig.savefig(vis_dir.joinpath('b_{}.png'.format(index)))
        self.vis_asm_components(self.config.asm_dir)

    def vis_asm_components(self, asm_dir):
        '''
        可视化ASM的不同components
        asm_dir: 保存asm的dir，相当于proj_dir，但是这里可以改
        '''
        pts_num = self.config.total_pts_num
        model = pickle.load(open(asm_dir.joinpath('model'), 'rb'))
        save_dir = asm_dir.joinpath('PCA vis')
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        # for i in range(model.eig_vecs.shape[0]):
        #     print(str(i + 1), ':', torch.sum(model.eig_vals[: i + 1]) / torch.sum(model.eig_vals))
        for i in range(model.eig_vecs.shape[0]):  # 可视化
            mode_minus3 = -math.sqrt(3) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_minus2 = -math.sqrt(2) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_minus1 = -1 * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus1 = 1 * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus2 = math.sqrt(2) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus3 = math.sqrt(3) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            plt.figure(figsize=(77, 11))
            ax1 = plt.subplot(171)
            ax1.invert_yaxis()
            plt.plot(mode_minus3[:, 0], mode_minus3[:, 1], 'r')
            plt.subplot(172, sharex=ax1, sharey=ax1)
            plt.plot(mode_minus2[:, 0], mode_minus2[:, 1], 'r')
            plt.subplot(173, sharex=ax1, sharey=ax1)
            plt.plot(mode_minus1[:, 0], mode_minus1[:, 1], 'r')
            plt.subplot(174, sharex=ax1, sharey=ax1)
            plt.plot(model.mean_shape[:, 0], model.mean_shape[:, 1], 'r')
            plt.subplot(175, sharex=ax1, sharey=ax1)
            plt.plot(mode_plus1[:, 0], mode_plus1[:, 1], 'r')
            plt.subplot(176, sharex=ax1, sharey=ax1)
            plt.plot(mode_plus2[:, 0], mode_plus2[:, 1], 'r')
            plt.subplot(177, sharex=ax1, sharey=ax1)
            plt.plot(mode_plus3[:, 0], mode_plus3[:, 1], 'r')
            plt.suptitle('PCA Mode ' + str(i) + '   -sqrt(3)-->+sqrt(3)', fontsize=28)
            plt.savefig(save_dir.joinpath('PCA_Mode_' + str(i) + '.png'))
            plt.cla()
            plt.close('all')

    def kmeans_cluster(self, k=3):
        '''
        数据集预处理：用全局ASM无约束计算各个样本的坐标，用于聚类
        '''
        masm_path = self.proj_dir.parent.joinpath('{}_clusters'.format(k))
        coordinates = pd.read_csv(self.proj_dir.joinpath('labeled_b.csv'), index_col=0)
        model = pickle.load(open(self.proj_dir.joinpath('model'), 'rb'))
        weight = model.eig_vals / torch.sum(model.eig_vals)
        weighted_coordinate = coordinates * weight
        kmeans = KMeans(k).fit(weighted_coordinate)
        # kmeans聚类，分开样本
        for cls in range(k):
            samples = {}
            samples['name'] = []
            cls_save_dir = masm_path.joinpath(str(cls))
            if not cls_save_dir.exists():
                cls_save_dir.mkdir(parents=True)
            for index, label in zip(coordinates.index, kmeans.labels_):
                if label == cls:
                    samples['name'].append(index)
            # print('cls:{} samples:{}'.format(cls, len(samples['name'])))
            # if len(samples['name']) == 1:
            #     print(samples['name'])
            #     print(index)
            pd.DataFrame(samples).to_csv(cls_save_dir.joinpath('samples.csv'))
            # 对聚类后的样本建模
            samples_pts = {k: self.dataset_pts[k] for k in samples['name']}
            samples_pts_tensor = torch.Tensor([v for v in samples_pts.values()])  # [N, 51, 2]
            masm_model = ActiveShapeModel(samples_pts_tensor)
            masm_model.save(cls_save_dir.joinpath('model'))
            self.vis_asm_components(cls_save_dir)


def main(config) -> None:
    processer = ASM_process(config)
    # Step1: train DSM之前，先对data建模，asm_save_dir是config里默认的
    processer.fit_transform_labeled()  # 1_clusters
    # for k in range(2, 31):
    # for k in list(range(2, 11)) + [15, 20, 25, 30]:
    #     processer.kmeans_cluster(k)
    # Step2: train DSM之后，根据DSM的预测结果，抽样每个pts，获取其位置及权重
    # TODO
    # Step3: unlabeled data 加进去，重新建模
    # TODO
    # Step4: 利用重新建模后的ASM，再训练DSM
    # TODO
