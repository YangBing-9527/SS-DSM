import math
import cv2
import json
import torch
import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA, IncrementalPCA

from models.utils import shape_to_iou, vis_pts_polylines


def similarity_transform_torch(S1, S2):
    '''
    S1: [N, 2] ---> S2: [N, 2]
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    assert S1.shape[0] == S2.shape[0]  # N=N
    S1 = S1.T
    S2 = S2.T
    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)  # [2, 1]
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1  # [2, N]
    X2 = S2 - mu2
    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2)  # 1个数
    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)  # [2, 2]
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)  # [2, 2]
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)  # [2, 2]
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))  # [2, 2]
    # Construct R.
    R = V.mm(Z.mm(U.T))  # [2, 2]
    # 5. Recover scale.
    # scale = torch.trace(R.mm(K)) / var1  # 1个数
    scale = torch.trace(R.mm(K)) / var1  # 1个数
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))  # [2, 1]
    # 7. Error
    S1_hat = scale * R.mm(S1) + t  # [2, N]
    S1_hat = S1_hat.T  # [N, 2]
    return S1_hat


def batch_similarity_transform_torch(S1, S2):
    '''
    S1:[B, N, 2] ---> S2: [B, N, 2]
    没有归一化，直接SVD分解的
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    assert len(S1.shape) == 3
    assert S1.shape[-1] == 2
    assert S2.shape[1] == S1.shape[1], 'the number of points in S1 and S2 are different'

    S1 = S1.permute(0, 2, 1)  # [B, 2, N]
    S2 = S2.permute(0, 2, 1)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)  # [B, 2, 1]
    mu2 = S2.mean(axis=-1, keepdims=True)  # [B, 2, 1]
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def perspective_transform_np(pts4, shape, pt_idxes):
    '''
    根据pts4，求透视变换矩阵，将shape映射到新位置
    '''
    shape_4pt = shape[pt_idxes]
    mtx = cv2.getPerspectiveTransform(shape_4pt.astype(np.float32), pts4.astype(np.float32)).astype(np.float32)  # [3, 3]
    ones = np.ones((shape.shape[0], 1))  # [51, 1]
    shape_ones = np.concatenate((shape, ones), axis=1)  # [51, 3]
    pos_shape = shape_ones @ mtx.T  # [51, 3]
    w = np.expand_dims(pos_shape[:, -1], axis=1).repeat(2, axis=1)  # [51, 2]
    shape = pos_shape[:, :2] / w  # [51, 2]
    return shape


def affine_transform_np(pts3, shape, pt_idxes):
    '''
    根据pts3，求仿射变换矩阵，将shape映射到新位置
    '''
    shape_3pt = shape[pt_idxes]
    mtx = cv2.getAffineTransform(shape_3pt.astype(np.float32), pts3.astype(np.float32)).astype(np.float32)  # [2, 3]
    ones = np.ones((shape.shape[0], 1))  # [51, 1]
    shape_ones = np.concatenate((shape, ones), axis=1)  # [51, 3]
    pos_shape = shape_ones @ mtx.T  # [51, 3]
    # w = np.expand_dims(pos_shape[:, -1], axis=1).repeat(2, axis=1)  # [51, 2]
    shape = pos_shape[:, :2]  # / w  # [51, 2]
    return shape


def pts_transform_np(pts, pt_idxes, mean_shape):
    '''
    调用上面两种transoform，init shape
    inputs are all np
    '''
    # assert len(pts.shape) == 3, 'pts.shape != [B, N, 2]'
    trans_func = perspective_transform_np if len(pt_idxes) == 4 else affine_transform_np
    shape = trans_func(pts, mean_shape, pt_idxes)
    return shape


class ActiveShapeModel:
    def __init__(self, data, weights=None) -> None:
        '''
        data: tensor, [N, 51, 2]
        '''
        self.fit(data, weights=None)

    def save(self, save_path):
        '''
        建模好的ASM，保存到save_path
        '''
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, data, weights=None):
        '''
        data: tensor, [N, 51, 2]
        weights: sample pts weight, [N, 51, 2]
        在给定data上训练ASM model
        '''
        # step1: procrustes
        sample_num, pts_num = data.shape[:2]
        procrusted = deepcopy(data)
        for t in range(10):  # 迭代若干次，对齐
            mean_sample = torch.mean(procrusted, axis=0)  # [N, 2]
            mu = torch.mean(mean_sample, dim=0, keepdim=True)
            var = torch.linalg.norm(mean_sample - mu)
            mean_sample /= var
            for i, sample in enumerate(data):
                mtx = similarity_transform_torch(sample, mean_sample)
                procrusted[i] = mtx
        if weights:
            procrusted = procrusted * weights
        mean_shape = torch.mean(procrusted, axis=0)  # , keepdims=True)
        # 根据特征值选特征
        procrusted_1d = torch.reshape(procrusted, (procrusted.shape[0], -1))  # [N, 98]
        cov_mat = torch.cov(procrusted_1d.T)
        eig_val, eig_vec = torch.linalg.eigh(cov_mat)  # [98,], [98, 98] 从小到大排序
        eig_num = 10 if sample_num > 20 else sample_num // 2
        # eig_num = 5  # only for 0.1 labeled data
        eig_vecs = torch.zeros((eig_num, eig_vec.shape[0]))
        eig_vals = torch.zeros((eig_num))
        for i in range(eig_num):
            eig_vecs[i] = eig_vec[:, -i - 1]
            eig_vals[i] = eig_val[-i - 1]
        self.mean_shape = mean_shape
        self.eig_vecs = eig_vecs
        self.eig_vals = eig_vals

    def transform(self, sample, weights=None):
        '''
        将sample pts转变为b，再转回sample pts，修剪sample
        '''
        sample_mtx = similarity_transform_torch(sample, self.mean_shape)
        delta_mtx = sample_mtx - self.mean_shape  # [51, 2]
        if weights is not None:
            delta_mtx = weights * delta_mtx
        b = torch.matmul(self.eig_vecs, delta_mtx.reshape(-1))  # [10, 102]x[102]
        deform = deepcopy(self.mean_shape.reshape(-1))  # [102,]
        for i in range(self.eig_vecs.shape[0]):  # 10
            deform += self.eig_vecs[i] * b[i]
        pos = similarity_transform_torch(deform.reshape(-1, 2), sample)
        return b, pos

    def to(self, device):
        '''
        将ASM model转移到指定device
        '''
        self.mean_shape = self.mean_shape.to(device)
        self.eig_vecs = self.eig_vecs.to(device)
        self.eig_vals = self.eig_vals.to(device)
        return self


class ActiveShapeModel_sklearn:
    def __init__(self, config) -> None:
        self.config = config
        self.root = Path(config.dataset_path)
        self.asm_save_dir = Path(config.asm_save_dir)
        self.model_dir = self.asm_save_dir.joinpath('model dir')
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
        self.input_size = config.input_size
        self.vis_ssm = config.asm_vis_ssm
        if config.asm_init_model:  # 重新训练
            self.init_model()
        else:  # 加载现成的
            self.load_model()
            self.load_gaussion_bs()

    def init_model(self, save_model=True):
        '''
        重新训练 ASM model
        '''
        total_pts = json.load(open(self.root.joinpath('shape_544x736.json')))
        trainset = json.load(open(self.root.joinpath('SSL', 'split', str(self.config.labeled_ratio) + '.json')))
        labeled_names = trainset['labeled']
        labeled_pts_np = np.array([total_pts[v[:-4]] for v in labeled_names])  # [544, 51, 2]
        ssm = self.ssm(labeled_pts_np)
        if save_model:
            joblib.dump(ssm, self.model_dir.joinpath('ssm'))
        # self.load_model()
        self.get_trainset_b(total_pts, self.config.asm_max_bs)
        self.load_gaussion_bs()

    # def load_multi_model(self):
    #     '''
    #     加载MASM的mean shape
    #     '''
    #     model_dir = self.asm_save_dir.joinpath('model dir')
    #     self.mean_shape = np.load(model_dir.joinpath('mean_shape.npy'))  # [49, 2]

    # def load_model(self):
    #     '''
    #     加载已经训练好的 mean_shape/eig_vecs/eig_vals
    #     '''
    #     # 按numpy格式加载
    #     model_dir = self.asm_save_dir.joinpath('model dir')
    #     self.mean_shape = np.load(model_dir.joinpath('mean_shape.npy'))  # [49, 2]
    #     self.eig_vecs = np.load(model_dir.joinpath('eig_vecs.npy'))  # [10, 98]
    #     self.eig_vals = np.load(model_dir.joinpath('eig_vals.npy'))  # [10]
    #     self.max_bs = [3 * math.sqrt(abs(self.eig_vals[i])) for i in range(self.eig_vals.shape[0])]

    # def load_gaussion_bs(self):
    #     '''
    #     为shape model 加上 gaussion_b
    #     '''
    #     model_dir = self.asm_save_dir.joinpath('model dir')
    #     self.gaussion_b = np.load(model_dir.joinpath('gaussian_b.npy'))  # [2, 10]

    def procrustes_alignment(self, total_pts, procrustes_num=10):
        '''
        total_pts：需要对齐的样本，[N, 51, 2]
        procrustes_num: 对齐迭代次数
        '''
        W, H = self.input_size
        reversed = np.zeros_like(total_pts, dtype=float)  # 针对睫状肌左右朝向不一致
        # for i, pts in enumerate(total_pts):
        for i in range(total_pts.shape[0]):
            pts = total_pts[i]  # WH
            xs = pts[:, 0]
            ys = pts[:, 1]
            if pts[0, 0] < pts[20, 0]:  # 如果平坦部朝左，则沿竖向中轴线翻转坐标
                xs = W - xs
            trans = np.column_stack((xs, ys))
            reversed[i] = trans
        # procrustes
        procrusted = deepcopy(reversed)
        # if procrusted.shape[0] <=10:
        sample_num = total_pts.shape[0]
        if sample_num == 1:
            return procrusted.squeeze(0).astype(np.float32)
        else:
            for t in range(procrustes_num):  # 迭代若干次，对齐
                mean_sample = np.mean(procrusted, axis=0)
                for i, sample in enumerate(reversed):
                    mtx1, mtx2, disparity = procrustes(mean_sample, sample)
                    procrusted[i] = mtx2
            return procrusted

    def ssm(self, total_pts):
        '''
        statistic shape model
        形状对齐算法
        vis: 是否保存形状变化的可视化图
        '''
        vis_dir = self.asm_save_dir.joinpath('PCA vis')
        model_dir = self.asm_save_dir.joinpath('model dir')
        for d in [vis_dir, model_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        # step 1: procrustes analysis
        procrusted = self.procrustes_alignment(total_pts)
        pts_num = self.config.total_pts_num
        eig_num = 10 if ssm.n_samples_ > 20 else ssm.n_samples_ // 2
        ssm = PCA(n_components=eig_num)
        ssm.fit(procrusted.reshape(procrusted.shape[0], -1))
        if self.vis_ssm:
            for i in range(1, eig_num + 1):
                # print ssm的前eig_num个特征值
                print(i, ':', np.sum(ssm.explained_variance_ratio_[:i]))
                # 可视化前特征向量叠加mean shape后的变化
                mode_minus3 = -math.sqrt(3) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                mode_minus2 = -math.sqrt(2) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                mode_minus1 = -math.sqrt(1) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                mode_plus1 = math.sqrt(1) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                mode_plus2 = math.sqrt(2) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                mode_plus3 = math.sqrt(3) * ssm.singular_values_[i] ** 0.5 * ssm.n_components_[i].reshape((pts_num, 2)) + ssm.mean_
                plt.figure(figsize=(77, 11))
                ax1 = plt.subplot(171)
                ax1.invert_yaxis()
                plt.plot(mode_minus3[:, 0], mode_minus3[:, 1], 'r')
                plt.subplot(172, sharex=ax1, sharey=ax1)
                plt.plot(mode_minus2[:, 0], mode_minus2[:, 1], 'r')
                plt.subplot(173, sharex=ax1, sharey=ax1)
                plt.plot(mode_minus1[:, 0], mode_minus1[:, 1], 'r')
                plt.subplot(174, sharex=ax1, sharey=ax1)
                plt.plot(ssm.mean_[:, 0], ssm.mean_[:, 1], 'r')
                plt.subplot(175, sharex=ax1, sharey=ax1)
                plt.plot(mode_plus1[:, 0], mode_plus1[:, 1], 'r')
                plt.subplot(176, sharex=ax1, sharey=ax1)
                plt.plot(mode_plus2[:, 0], mode_plus2[:, 1], 'r')
                plt.subplot(177, sharex=ax1, sharey=ax1)
                plt.plot(mode_plus3[:, 0], mode_plus3[:, 1], 'r')
                plt.suptitle('PCA Mode ' + str(i) + '   -sqrt(3)-->+sqrt(3)', fontsize=28)
                plt.savefig(vis_dir.joinpath('PCA_Mode_' + str(i) + '.png'))
                plt.cla()
        return ssm

    def get_trainset_b(self, total_pts, max_bs=True):
        '''
        获取训练集所有样本的 b，保存为 csv
        max_bs: 是否约束 b
        '''
        img_dir = self.root.joinpath('image_544x736')
        imgs = sorted([x for x in img_dir.iterdir()])
        imgPath = imgs[: int(0.7 * len(imgs))]
        mode_name = 'max_bs' if max_bs else 'bs'
        save_dir = self.asm_save_dir.joinpath('check_label', mode_name)
        model_dir = self.asm_save_dir.joinpath('model dir')
        for d in [save_dir]:
            if not d.exists():
                d.mkdir(parents=True)
        label_dir = self.root.joinpath('label_544x736')
        ious = 0
        b_resuls = {}
        iou_resuls = {}
        ssm = joblib.load(self.asm_save_dir.joinpath('ssm'))
        for img_path in imgPath:
            # img_path = imgPath[index]
            image = cv2.imread(str(img_path))
            image = image / np.max(image)  # [H, W]
            label_path = label_dir.joinpath(img_path.name)
            label = cv2.imread(str(label_path), -1)
            pts = np.array(total_pts[img_path.stem])  # [51, 2], HW
            # step 1：归一化后，普式对齐，统一到相同的scale
            pts_mean = np.mean(pts, 0)
            pts_var = np.linalg.norm(pts - pts_mean)
            pts_norm = (pts - pts_mean) / pts_var
            mean_shape_mtx, pts_mtx, disparity = procrustes(ssm.mean_, pts)
            # step 2：ssm transform
            b = ssm.transform(pts_mtx)
            deform = ssm.inverse_transform(b)
            pts_norm, deform, disparity = procrustes(pts_norm, deform.reshape(-1, 2))
            pos = deform * pts_var + pts_mean  # 用pts的变化来恢复pred
            iou = shape_to_iou(pos, label // 255)
            vis_pts_polylines(
                pts=pos, WH=self.input_size, image=image, label=label, title='iou: {:.4f}'.format(iou), save_path=save_dir.joinpath(img_path.name)
            )
            iou_resuls[img_path.stem] = iou
            b_resuls[img_path.stem] = b
            ious += iou
            # break
        iou_resuls['avg_iou'] = ious / len(imgPath)
        iou_resuls = pd.DataFrame.from_dict(iou_resuls, orient='index')
        iou_resuls.to_csv(save_dir.joinpath('iou.csv'))  # _no_cons
        b_resuls = pd.DataFrame.from_dict(b_resuls, orient='index')
        b_resuls.to_csv(save_dir.joinpath('b.csv'))
        mius = []
        vars = []
        for index in range(len(b_resuls.columns)):
            ax = b_resuls[index]
            fig = sns.displot(ax, fit=norm, kde=True, fit_kws={'label': 'Norm'}, kde_kws={'label': 'KDE'}, label='Count')
            plt.legend()
            miu, std = norm.fit(ax)
            mius.append(miu)
            vars.append(std)
            sns_fig = fig.get_figure()
            sns_fig.savefig(save_dir.joinpath('b_{}.png'.format(index)))
            sns_fig.clear()
        # gaussion b
        gaussian_b = np.array([mius, vars])  # [2, 10]
        print(gaussian_b)
        np.save(model_dir.joinpath('gaussian_b'), gaussian_b)


# if __name__ == '__main__':
#     from exps.configs.CAMUS.unet import Config

#     asm_process = ASM_process(Config())
#     # asm_process.fit_transform()
#     for k in range(2, 11):
#         asm_process.kmeans_cluster(k)
