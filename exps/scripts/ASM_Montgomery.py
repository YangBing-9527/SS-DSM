import cv2
import json
import torch
import pickle
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from shutil import copy
from pathlib import Path
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from models.ASM import ActiveShapeModel
from models.utils import shape_to_iou, shape_to_mask, vis_pts_polylines, soft_argmax_tensor


class NpEncoder(json.JSONEncoder):
    '''
    json encoder 专用
    '''

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


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
    gaussian_map = gamma / (2 * np.pi * sigmma**2) * np.exp(-0.5 * dist / (sigmma**2))
    # cv2.imwrite('gaussian.png', gaussian_map*255)
    return gaussian_map


def make_pts_heatmap_np(shape, WH, sigmma=3, gamma=7):
    '''
    制作点的热力图
    shape: [N, 2]
    '''
    shape = np.int32(shape)
    masks = np.zeros((shape.shape[0], WH[1], WH[0]), dtype=np.float32)
    for i in range(shape.shape[0]):
        pt = shape[i].astype(int)
        mask = gaussian(pt, WH[1], WH[0], sigmma, gamma)
        mask /= mask.max()
        masks[i] = mask
    return masks


class ASM_process:
    '''
    调用ASM，实现多种功能
    '''

    def __init__(self, config) -> None:
        self.config = config
        self.dataset_path = Path(self.config.dataset_path)
        self.dataset_pts = json.load(open(self.dataset_path.joinpath('shape.json')))
        self.proj_dir = Path(self.config.asm_dir)
        self.ss_proj_dir = Path(self.config.asm_plus_dir)
        self.imgWH = [x * self.config.resize_ratio for x in self.config.input_size]

    def fit_transform(self, data_weights=None):
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
        trainset_pts_tensor_L = []
        trainset_pts_tensor_R = []
        for v in trainset_pts.values():
            trainset_pts_tensor_L.append(v['L'])
            trainset_pts_tensor_R.append(v['R'])
        trainset_pts_tensor_L = torch.Tensor(trainset_pts_tensor_L)  # [N, 51, 2]
        trainset_pts_tensor_R = torch.Tensor(trainset_pts_tensor_R)  # [N, 51, 2]
        model_L = ActiveShapeModel(trainset_pts_tensor_L, data_weights)
        model_R = ActiveShapeModel(trainset_pts_tensor_R, data_weights)
        model_L.save(self.proj_dir.joinpath('model_L'))
        model_R.save(self.proj_dir.joinpath('model_R'))

        # Part II: transform
        img_dir = self.dataset_path.joinpath('image')
        label_dir = self.dataset_path.joinpath('label_single')
        trainset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', str(self.config.labeled_ratio) + '.json')))
        testset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', 'test.json')))['test']
        labeled_names = trainset_names['labeled']
        d = {'trainset': labeled_names, 'testset': testset_names}
        for key, names in d.items():
            vis_dir = self.proj_dir.joinpath('check_label', key)
            if not vis_dir.exists():
                vis_dir.mkdir(parents=True)
            ious_L = 0
            ious_R = 0
            b_resuls_L = {}
            b_resuls_R = {}
            iou_resuls_L = {}
            iou_resuls_R = {}
            count = 0
            for img_name in names:
                img_path = img_dir.joinpath(img_name)
                image = cv2.imread(str(img_path))
                image = image / np.max(image)  # [H, W]
                # 区分左右
                if self.dataset_pts[img_name[:-4]]['L'] is not None:
                    label_L_path = label_dir.joinpath(img_name[:-4] + '_L.png')
                    label_L = cv2.imread(str(label_L_path), cv2.IMREAD_GRAYSCALE)
                    pts = torch.Tensor(self.dataset_pts[img_name[:-4]]['L'])  # [51, 2], HW
                    # pts[:, 0] = self.config.input_size[0] - pts[:, 0]  # 左右翻转
                    b, shape = model_L.transform(pts)
                    # shape[:, 0] = self.config.input_size[0] - pts[:, 0]
                    iou = shape_to_iou(shape, label_L // 255)
                    vis_pts_polylines(
                        pts=shape,
                        WH=self.imgWH,
                        image=image,
                        label=label_L,
                        title='iou: {:.4f}'.format(iou),
                        save_path=vis_dir.joinpath(label_L_path.name),
                    )
                    iou_resuls_L[img_path.stem + '_L'] = iou
                    b_resuls_L[img_path.stem + '_L'] = b.tolist()
                    ious_L += iou
                    count += 1
                if self.dataset_pts[img_name[:-4]]['R'] is not None:
                    label_R_path = label_dir.joinpath(img_name[:-4] + '_R.png')
                    label_R = cv2.imread(str(label_R_path), cv2.IMREAD_GRAYSCALE)
                    pts = torch.Tensor(self.dataset_pts[img_name[:-4]]['R'])
                    b, shape = model_R.transform(pts)
                    iou = shape_to_iou(shape, label_R // 255)
                    vis_pts_polylines(
                        pts=shape,
                        WH=self.imgWH,
                        image=image,
                        label=label_R,
                        title='iou: {:.4f}'.format(iou),
                        save_path=vis_dir.joinpath(label_R_path.name),
                    )
                    iou_resuls_R[img_path.stem + '_R'] = iou
                    b_resuls_R[img_path.stem + '_R'] = b.tolist()
                    ious_R += iou
                    count += 1
                # break
            iou_resuls_L['avg_iou'] = ious_L / count
            iou_resuls_R['avg_iou'] = ious_R / count
            iou_resuls_L = pd.DataFrame.from_dict(iou_resuls_L, orient='index')
            iou_resuls_L.to_csv(vis_dir.joinpath('iou_L.csv'))  # _no_cons
            iou_resuls_R = pd.DataFrame.from_dict(iou_resuls_R, orient='index')
            iou_resuls_R.to_csv(vis_dir.joinpath('iou_R.csv'))  # _no_cons
            b_resuls_pd_L = pd.DataFrame.from_dict(b_resuls_L, orient='index')
            b_resuls_pd_L.to_csv(self.proj_dir.joinpath('{}_b_L.csv'.format(key)))
            b_resuls_pd_R = pd.DataFrame.from_dict(b_resuls_R, orient='index')
            b_resuls_pd_R.to_csv(self.proj_dir.joinpath('{}_b_R.csv'.format(key)))
            for index in range(len(b_resuls_pd_L.columns)):
                ax = b_resuls_pd_L[index]
                fig = sns.displot(ax, kde=True)
                fig.savefig(vis_dir.joinpath('b_L_{}.png'.format(index)))
            for index in range(len(b_resuls_pd_R.columns)):
                ax = b_resuls_pd_R[index]
                fig = sns.displot(ax, kde=True)
                fig.savefig(vis_dir.joinpath('b_R_{}.png'.format(index)))
        self.vis_asm_components(self.config.asm_dir, 'L')
        self.vis_asm_components(self.config.asm_dir, 'R')

    def vis_asm_components(self, asm_dir, part='L'):
        '''
        可视化ASM的不同components
        asm_dir: 保存asm的dir，相当于proj_dir，但是这里可以改
        '''
        pts_num = self.config.total_pts_num // 2
        model_name = 'model_' + part
        model = pickle.load(open(asm_dir.joinpath(model_name), 'rb'))
        save_dir = asm_dir.joinpath('PCA vis', part)
        save_dir.mkdir(parents=True, exist_ok=True)
        # if not save_dir.exists():
        #     save_dir.mkdir(parents=True)
        # for i in range(model.eig_vecs.shape[0]):
        #     print(str(i + 1), ':', torch.sum(model.eig_vals[: i + 1]) / torch.sum(model.eig_vals))
        for i in range(model.eig_vecs.shape[0]):  # 可视化
            mode_minus3 = -np.sqrt(3) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_minus2 = -np.sqrt(2) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_minus1 = -1 * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus1 = 1 * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus2 = np.sqrt(2) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
            mode_plus3 = np.sqrt(3) * model.eig_vals[i] ** 0.5 * model.eig_vecs[i, :].reshape((pts_num, 2)) + model.mean_shape
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

    def kmeans_cluster(self, part='L', k=3):
        '''
        数据集预处理：用全局ASM无约束计算各个样本的坐标，用于聚类
        '''
        # 从self.dataset_pts中提取指定part的点集
        dataset_single_pts = {}
        for name, parts in self.dataset_pts.items():
            if parts[part]:  # 如果该part存在点集
                dataset_single_pts[name + '_' + part] = parts[part]
        
        masm_path = self.proj_dir.parent.joinpath('{}_clusters'.format(k), part)
        coordinates = pd.read_csv(self.proj_dir.joinpath('labeled_b_'+part+'.csv'), index_col=0)
        model = pickle.load(open(self.proj_dir.joinpath('model_'+part), 'rb'))
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
            pd.DataFrame(samples).to_csv(cls_save_dir.joinpath('samples.csv'))
            
            # 对聚类后的样本建模
            samples_pts = {k: dataset_single_pts[k] for k in samples['name']}
            samples_pts_tensor = torch.Tensor([v for v in samples_pts.values()])  # [N, 70, 2]
            masm_model = ActiveShapeModel(samples_pts_tensor)
            masm_model.save(cls_save_dir.joinpath('model'))
            self.vis_asm_components(cls_save_dir, part)

    def select_psudo_label(self, heatmap_dir, bbox_margin=25, squrt_factor=5, threshold=0.5):
        '''
        Step1: 根据DSM预测关键点热力图，计算每个关键点的权重，用于ASM建模
        Step2: ASM建模，生成新的ASM模型，以及配套的关键点坐标、b值
        Step3：根据标准筛选伪标签
        '''
        heatmap_dir = self.config.save_path.joinpath('gen_unlabeled', 'heatmap_tensor')
        model = pickle.load(open(self.proj_dir.joinpath('model'), 'rb'))
        shapes = {}
        weights = {}
        bs_range = torch.sqrt(squrt_factor * model.eig_vals)  # [10]
        bs_range = bs_range.repeat(2)  # [20]

        def get_diff(model, tensor_np, shape, bbox_margin, LR):
            psudo_labels = make_pts_heatmap_np(shape, [tensor.shape[2], tensor.shape[1]], 3, 7)  # [51, H/2, W/2]
            diffs = np.zeros((shape.shape[0]))  # [102]
            for i in range(shape.shape[0]):
                pt = shape[i]
                bbox = [
                    max(0, pt[0] - bbox_margin),
                    max(0, pt[1] - bbox_margin),
                    min(tensor.shape[2], pt[0] + bbox_margin),
                    min(tensor.shape[1], pt[1] + bbox_margin),
                ]  # [L, U, R, D]
                diffs[i] = np.mean(np.abs(psudo_labels[i][bbox[0] : bbox[2], bbox[1] : bbox[3]] - tensor_np[i][bbox[0] : bbox[2], bbox[1] : bbox[3]]))
            diff = np.where(np.isnan(diffs), 100, diffs)
            diff_weight = np.exp(-diff * 10)
            # for idx in [0, 20, 55]:
            #     frame = tensor_np[idx] * 255
            #     pt = shape[idx]
            #     L = max(0, pt[0] - bbox_margin)
            #     U = max(0, pt[1] - bbox_margin)
            #     R = min(tensor.shape[2], pt[0] + bbox_margin)
            #     D = min(tensor.shape[1], pt[1] + bbox_margin)
            #     frame = cv2.rectangle(frame, (L, U), (R, D), 255, 2)
            #     cv2.imwrite('pt_{}.png'.format(idx), frame)
            # 计算ASM变换后的位置差异，差异大的点权重小
            if LR == 'L':
                shape[:, 0] = self.config.input_size[0] - shape[:, 0]  # 左右翻转
            b, shape_transformed = model.transform(torch.Tensor(shape))
            dist = np.mean(np.abs(shape - shape_transformed.numpy()), axis=1)  # [61]
            dist_weight = np.exp(-dist)
            weight = diff_weight + dist_weight
            if LR == 'L':
                shape_transformed[:, 0] = self.config.input_size[0] - shape_transformed[:, 0]  # 再翻转回来
            return b, shape_transformed, weight

        for tensor_path in heatmap_dir.iterdir():
            flag = True
            shape_result = {}
            weight_result = {}
            tensor = torch.load(tensor_path)  # [61, H/2, W/2]
            shapeLR = soft_argmax_tensor(tensor.unsqueeze(0))  # [1, 102, 2] WH
            shapeLR = shapeLR.squeeze(0).numpy().astype(int)  # [102, 2] WH
            tensor_np = tensor.numpy()
            b_L, shape_L, weight_L = get_diff(model, tensor_np, shapeLR[:51], bbox_margin, 'L')
            b_R, shape_R, weight_R = get_diff(model, tensor_np, shapeLR[51:], bbox_margin, 'R')
            if np.concatenate([weight_L, weight_R]).mean() < threshold:
                flag = False
            for i, j in zip(np.concatenate([b_L, b_R]), bs_range):
                if i < -j or i > j:
                    flag = False
                    break
            if flag:
                name = tensor_path.stem[:-16]
                shape_result['L'] = shape_L.tolist()
                shape_result['R'] = shape_R.tolist()
                weight_result['L'] = weight_L.tolist()
                weight_result['R'] = weight_R.tolist()
                shapes[name] = shape_result
                weights[name] = weight_result
            # break
        with open(heatmap_dir.parent.joinpath('selected_psudo_shapes.json'), 'w') as f:
            json.dump(shapes, f, cls=NpEncoder)
        with open(heatmap_dir.parent.joinpath('selected_psudo_weights.json'), 'w') as f:
            json.dump(weights, f, cls=NpEncoder)

    def make_psudo_label(self):
        '''
        根据selected_psudo_label制作分割结果、热力图、shape伪标签，合并labeled GT组成完整的trainset
        '''
        model = pickle.load(open(self.ss_proj_dir.joinpath('model'), 'rb'))
        split_json = json.load(open(self.dataset_path.joinpath('SSL', 'split', str(self.config.labeled_ratio) + '.json')))
        testset_names = json.load(open(self.dataset_path.joinpath('SSL', 'split', 'test.json')))['test']
        selected_psudo_shapes = json.load(open(self.config.save_path.joinpath('gen_unlabeled', 'selected_psudo_shapes.json')))
        selected_psudo_weights = json.load(open(self.config.save_path.joinpath('gen_unlabeled', 'selected_psudo_weights.json')))
        selected_psudo_names = [x + '.png' for x in selected_psudo_shapes.keys()]
        # 合并trainset_b和testset_b
        trainset_b = pd.read_csv(self.ss_proj_dir.joinpath('trainset_b.csv'), index_col=0)
        testset_b = pd.read_csv(self.ss_proj_dir.joinpath('testset_b.csv'), index_col=0)
        b = pd.concat([trainset_b, testset_b])
        b.to_csv(self.ss_proj_dir.joinpath('b.csv'))
        # 为unlabeled data制作伪标签
        mask_save_dir = self.config.save_path.joinpath('psudo_label', 'label_544x736')
        heatmap_save_dir = self.config.save_path.joinpath('psudo_label', 'heatmap', self.config.heatmap_dir_name)
        for d in [mask_save_dir, heatmap_save_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        shapes = {}
        for name in selected_psudo_names:
            shape_pred = torch.Tensor(selected_psudo_shapes[name[:-4]])
            shape_weight = torch.Tensor(selected_psudo_weights[name[:-4]])
            shape = model.transform(shape_pred, shape_weight.view(-1, 1))[1]
            shape = shape * self.config.resize_ratio
            shapes[name[:-4]] = shape.numpy().astype(np.int32).tolist()
            mask = shape_to_mask(shape, self.imgWH)
            cv2.imwrite(str(mask_save_dir.joinpath(name)), mask * 255)
            heatmap = make_pts_heatmap_np(shape, self.imgWH, 3, 7)
            heatmap_tensor = torch.tensor(heatmap)
            torch.save(heatmap_tensor, heatmap_save_dir.joinpath(name[:-4] + '.pt'))
            # break
        # 从labeled data、testset的GT copy过来
        total_shapes = json.load(open(self.dataset_path.joinpath('shape_544x736.json')))
        for name in split_json['labeled'] + testset_names:
            mask_path = self.dataset_path.joinpath('label_544x736', name)
            dst_mask_path = mask_save_dir.joinpath(name)
            copy(str(mask_path), str(dst_mask_path))
            heamap_path = self.dataset_path.joinpath('heatmap', self.config.heatmap_dir_name, name[:-4] + '.pt')
            dst_heatmap_path = heatmap_save_dir.joinpath(name[:-4] + '.pt')
            copy(str(heamap_path), str(dst_heatmap_path))
            shapes[name[:-4]] = total_shapes[name[:-4]]
        with open(mask_save_dir.parent.joinpath('shape_544x736.json'), 'w') as f:
            json.dump(shapes, f, cls=NpEncoder)


def main(config) -> None:
    processer = ASM_process(config)
    # Step1: train DSM之前，先对data建模，asm_save_dir是config里默认的
    processer.fit_transform()  # 1_clusters
    # for part in ['L', 'R']:
    #     processer.kmeans_cluster(2, part)

    # Step2: train DSM之后，根据DSM的预测结果，抽样每个pts，获取其位置及权重
    # processer.select_psudo_label(config.save_path.joinpath('gen_unlabeled'))
    # processer.fit_trainset()  # labeled和unlabeled数据集合并，重新建模
    # processer.transform_trainset()  # 检查re-ASM的效果，生成新的b值
    # processer.make_psudo_label()  # 根据re-ASM的结果，生成伪标签
    # processer.kmeans_cluster(config.asm_plus_dir, 10)
