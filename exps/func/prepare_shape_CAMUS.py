import cv2
import json
import math
import torch
import random
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm
from shutil import copy
from pathlib import Path
from skimage import filters
from sklearn.cluster import KMeans


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


def copy_to_dataset(root, dest):
    '''
    将root文件夹的 lvendo 的image、label复制到dest
    '''
    root = Path(root)
    dest = Path(dest)
    image_folder = dest.joinpath('image')
    label_folder = dest.joinpath('label')
    # 检查路径是否存在，不存在就创建
    for d in [image_folder, label_folder]:
        if not d.exists():
            d.mkdir(parents=True)

    for folder in root.iterdir():
        image_2CH_ED_path = folder.joinpath(folder.name + '_2CH_ED.png')
        image_2CH_ES_path = folder.joinpath(folder.name + '_2CH_ES.png')
        image_4CH_ED_path = folder.joinpath(folder.name + '_4CH_ED.png')
        image_4CH_ES_path = folder.joinpath(folder.name + '_4CH_ES.png')

        label_2CH_ED_path = folder.joinpath(image_2CH_ED_path.stem + '_gt_lvendo.png')
        label_2CH_ES_path = folder.joinpath(image_2CH_ES_path.stem + '_gt_lvendo.png')
        label_4CH_ED_path = folder.joinpath(image_4CH_ED_path.stem + '_gt_lvendo.png')
        label_4CH_ES_path = folder.joinpath(image_4CH_ES_path.stem + '_gt_lvendo.png')

        # image_2CH_ED_json_path = folder.joinpath(image_2CH_ED_path.stem + 'points.json')
        # image_2CH_ES_json_path = folder.joinpath(image_2CH_ES_path.stem + 'points.json')
        # image_4CH_ED_json_path = folder.joinpath(image_4CH_ED_path.stem + 'points.json')
        # image_4CH_ES_json_path = folder.joinpath(image_4CH_ES_path.stem + 'points.json')

        dest_image_2CH_ED_path = image_folder.joinpath(image_2CH_ED_path.name)
        dest_image_2CH_ES_path = image_folder.joinpath(image_2CH_ES_path.name)
        dest_image_4CH_ED_path = image_folder.joinpath(image_4CH_ED_path.name)
        dest_image_4CH_ES_path = image_folder.joinpath(image_4CH_ES_path.name)

        dest_label_2CH_ED_path = label_folder.joinpath(image_2CH_ED_path.name)
        dest_label_2CH_ES_path = label_folder.joinpath(image_2CH_ES_path.name)
        dest_label_4CH_ED_path = label_folder.joinpath(image_4CH_ED_path.name)
        dest_label_4CH_ES_path = label_folder.joinpath(image_4CH_ES_path.name)

        copy(image_2CH_ED_path, dest_image_2CH_ED_path)
        copy(image_2CH_ES_path, dest_image_2CH_ES_path)
        copy(image_4CH_ED_path, dest_image_4CH_ED_path)
        copy(image_4CH_ES_path, dest_image_4CH_ES_path)

        label_lst = [label_2CH_ED_path, label_2CH_ES_path, label_4CH_ED_path, label_4CH_ES_path]
        dest_lst = [dest_label_2CH_ED_path, dest_label_2CH_ES_path, dest_label_4CH_ED_path, dest_label_4CH_ES_path]
        for label_path, dest_label_path in zip(label_lst, dest_lst):
            label_2CH_ED = cv2.imread(str(label_path), -1)
            label_2CH_ED = np.where(label_2CH_ED > 0, 255, 0)
            cv2.imwrite(str(dest_label_path), label_2CH_ED)
        # break


def get_landmarks(root, dest, left_top_pts_num=30, top_right_pts_num=30):
    '''
    获取 Top、Left、Right关键点，并采样，shape顺序为：left->top->right
    root: dataset
    '''
    root = Path(root)
    dest = Path(dest)
    label_dir = dest.joinpath('label')
    total = {}  # 存放整个数据集
    count = 0
    for label_path in label_dir.iterdir():
        patient = label_path.stem.split('_')[0]
        json_path = root.joinpath(patient, label_path.stem + 'points.json')
        with open(json_path) as f:
            landmarks = json.load(f)  # [W, H]
        left = landmarks['left']
        top = landmarks['top']
        right = landmarks['right']
        left_seg_len = (top[1] - left[1]) / left_top_pts_num
        left_line = [int(left[1] + i * left_seg_len) for i in range(left_top_pts_num)]
        right_seg_len = (right[1] - top[1]) / top_right_pts_num
        right_line = [int(top[1] + i * right_seg_len) for i in range(top_right_pts_num)]

        label = cv2.imread(str(label_path), -1)  # [H, W]
        # 找left_line的点
        temp = []
        for pt_y in left_line:
            candidate_x = np.where(label[pt_y, :] > 100)
            temp.append(candidate_x[0][0])
        left_pts = [[i, j] for i, j in zip(temp, left_line)]
        # 找right_line的点
        temp = []
        for pt_y in right_line:
            candidate_x = np.where(label[pt_y, :] > 100)
            temp.append(candidate_x[0][-1])
        right_pts = [[i, j] for i, j in zip(temp, right_line)]
        shape = left_pts + right_pts + [list(map(int, right))]
        total[label_path.stem] = shape
        count += 1
        print(count)
    save_shape_path = dest.joinpath('shape.json')
    with open(save_shape_path, 'w') as f:
        f.write(json.dumps(total, cls=NpEncoder))


def resize_dataset_shapes(dest, size=(544, 736)):
    '''
    原图尺寸不一，统一到size，并创建对应的shape json
    '''
    dest = Path(dest)
    shapes = json.load(open(dest.joinpath('shape.json')))
    new_shapes = {}
    image_dir = dest.joinpath('image')
    label_dir = dest.joinpath('label')
    dest_image_dir = dest.joinpath('image_{}x{}'.format(size[0], size[1]))
    dest_label_dir = dest.joinpath('label_{}x{}'.format(size[0], size[1]))
    vis_image_dir = dest.joinpath('vis_{}x{}'.format(size[0], size[1]))
    for d in [dest_image_dir, dest_label_dir, vis_image_dir]:
        if not d.exists():
            d.mkdir(parents=True)
    for image_path in image_dir.iterdir():
        image = cv2.imread(str(image_path), -1)
        image_resized = cv2.resize(image, size)
        cv2.imwrite(str(dest_image_dir.joinpath(image_path.name)), image_resized)

        label_path = label_dir.joinpath(image_path.name)
        label = cv2.imread(str(label_path), -1)
        label_resized = cv2.resize(label, size)
        cv2.imwrite(str(dest_label_dir.joinpath(label_path.name)), label_resized)

        scale_mtx = np.array([[size[0] / image.shape[1], 0], [0, size[1] / image.shape[0]]])
        shape = shapes[image_path.stem]
        new_shape = np.array(shape) @ scale_mtx
        new_shapes[image_path.stem] = new_shape.astype(int)

        # for vis
        image_vis = cv2.polylines(image_resized, [new_shape.astype(int)], isClosed=True, color=255, thickness=2)
        cv2.imwrite(str(vis_image_dir.joinpath(image_path.name)), image_vis)
        # break
    save_path = dest.joinpath('shape_{}x{}.json'.format(size[0], size[1]))
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_shapes, cls=NpEncoder))


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


def make_heatmap(root, pt_indexes, WH, sigmma, gamma):
    '''
    制作点的热力图
    '''
    root = Path(root)
    save_path = root.joinpath('heatmap')
    vis_path = save_path.joinpath('vis_' + str(len(pt_indexes)) + '_' + str(sigmma))
    if not vis_path.exists():
        vis_path.mkdir(parents=True)
    tensor_path = save_path.joinpath('tensor_' + str(len(pt_indexes)) + '_' + str(sigmma))
    if not tensor_path.exists():
        tensor_path.mkdir(parents=True)
    points = json.load(open(root.joinpath('shape_544x736.json')))
    for item in tqdm(points.keys()):
        pt_lst = points[item]
        pts_np = np.array(pt_lst)
        masks = np.zeros((len(pt_indexes), WH[1], WH[0]), dtype=np.float32)
        for i in range(len(pt_indexes)):
            pt_index = pt_indexes[i]
            mask = gaussian(pts_np[pt_index], WH[1], WH[0], sigmma, gamma)
            masks[i] = mask
            # 可视化，手动检查
            factor = 255 / np.max(mask)
            cv2.imwrite(str(vis_path.joinpath(item + '_' + str(pt_index) + '.png')), mask * factor)
        for i in range(masks.shape[0]):
            factor = 1 / np.max(masks[i])
            masks[i] *= factor
        masks_tensor = torch.from_numpy(masks).type(torch.float32)
        torch.save(masks_tensor, tensor_path.joinpath(item + '.pt'))
        # break


def make_grad(root, k=25):
    root = Path(root)
    image_dir = root.joinpath('image_544x736')
    save_dir = root.joinpath('grad', 'laplacian_k' + str(k))
    # save_dir = root.joinpath('grad', 'Canny_' + str(k))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    for image_path in image_dir.iterdir():
        # print(image_path.name)
        image = cv2.imread(str(image_path))
        image = np.average(image, -1) / 255
        image = cv2.blur(image, (k, 1))
        grad = cv2.Laplacian(image, -1, ksize=k)
        grad = -grad

        # image = cv2.medianBlur(image, 5)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # grad = cv2.Canny(image, 100, 150)
        # scale = 255/np.max(image)
        # cv2.imwrite('a.png', image*scale)
        # sobelx = cv2.Sobel(image, -1, 1, 0, ksize=7)
        # sobely = cv2.Sobel(image, -1, 0, 1, ksize=7)
        # grad = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # grad = cv2.convertScaleAbs(grad)
        # grad = np.hstack((image, grad, gradAbs))
        # grad = np.average(grad, -1)
        # grad = np.abs(grad)
        scale = 255 / np.max(grad)
        # scale = 1
        # print(grad.shape)
        save_name = save_dir.joinpath(image_path.name)
        cv2.imwrite(str(save_name), grad * scale)
        break


def make_grad_heatmap(root, k, sigmma, gamma):
    '''
    继承自make_grad，用otsu将grad二值化，算距离图
    '''
    root = Path(root)
    grad_dir = root.joinpath('grad', 'laplacian_k{}'.format(k))
    binary_save_dir = root.joinpath('grad', 'grad_binary', 'laplacian_k{}'.format(k))
    heatmap_save_dir = root.joinpath('grad', 'grad_SDF', 'laplacian_k{}'.format(k))
    for b in [binary_save_dir, heatmap_save_dir]:
        if not b.exists():
            b.mkdir(parents=True)
    for img_path in grad_dir.iterdir():
        img = cv2.imread(str(img_path), -1)
        threshold = filters.threshold_otsu(img, nbins=256)
        img_binary = np.where(img > threshold, np.full_like(img, 255), np.full_like(img, 0.0))
        cv2.imwrite(str(binary_save_dir.joinpath(img_path.name)), img_binary)
        img_standard = img_binary // 255
        img_reversed = np.where(img_standard > 0, np.uint8(0), np.uint8(1))
        dist_trans_inner = cv2.distanceTransform(img_standard, cv2.DIST_L2, 5)
        dist_trans_outer = cv2.distanceTransform(img_reversed, cv2.DIST_L2, 5)
        dist_trans = dist_trans_inner + dist_trans_outer  # 内外都是正的
        gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist_trans / (sigmma**2))
        factor = 255 / np.max(gaussian_map)
        gaussian_map *= factor
        cv2.imwrite(str(heatmap_save_dir.joinpath(img_path.name)), gaussian_map)
        # break


def make_edge_heatmap(root, sigmma=2, gamma=7):
    '''
    计算label中各像素点到边缘的距离，结果转为tensor并保存
    epsilon：设定范围，太远的就不要了，否则数值太大。
    '''
    root = Path(root)
    label_dir = root.joinpath('label_544x736')
    save_dir = label_dir.parent.joinpath('SDF_tensor')
    tensor_dir = save_dir.joinpath('tensor_' + str(sigmma))
    vis_dir = save_dir.joinpath('vis_' + str(sigmma))
    for d in [tensor_dir, vis_dir]:
        if not d.exists():
            d.mkdir(parents=True)
    for label_path in label_dir.iterdir():
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label_reversed = np.where(label > 0, np.uint8(0), np.uint8(1))
        dist_trans_inner = cv2.distanceTransform(label, cv2.DIST_L2, 5)
        dist_trans_outer = cv2.distanceTransform(label_reversed, cv2.DIST_L2, 5)
        dist_trans = dist_trans_inner + dist_trans_outer  # 内外都是正的
        gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist_trans / (sigmma**2))
        # dist_trans = np.expand_dims(dist_trans, -1)
        # dist_trans = np.clip(dist_trans, -epsilon, +epsilon) / (epsilon*2)
        factor = 255 / np.max(gaussian_map)
        cv2.imwrite(str(vis_dir.joinpath(label_path.name)), gaussian_map * factor)
        factor = 1 / np.max(gaussian_map)
        tensor = torch.from_numpy(gaussian_map * factor).type(torch.float32)
        # tensor = tensor.permute(2, 0, 1)
        # tensor = tensor.unsqueeze(0)
        torch.save(tensor.unsqueeze(0), tensor_dir.joinpath(str(label_path.stem) + '.pt'))
        # break


def make_SDF(root):
    '''
    制作signed distance map, normalize sdf to [-1,1]```
    '''
    root = Path(root)
    label_dir = root.joinpath('label_544x736')
    save_dir = label_dir.parent.joinpath('SDF_tensor', 'distance')
    vis_dir = label_dir.parent.joinpath('SDF_tensor', 'vis')
    for d in [save_dir, vis_dir]:
        if not d.exists():
            d.mkdir(parents=True)
    for label_path in label_dir.iterdir():
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label_reversed = np.where(label > 0, np.uint8(0), np.uint8(1))
        dist_inner = cv2.distanceTransform(label, cv2.DIST_L2, 5)
        dist_inner_norm = (dist_inner - np.min(dist_inner)) / (np.max(dist_inner) - np.min(dist_inner))
        dist_outer = cv2.distanceTransform(label_reversed, cv2.DIST_L2, 5)
        dist_outer_norm = (dist_outer - np.min(dist_outer)) / (np.max(dist_outer) - np.min(dist_outer))
        dist_norm = dist_inner_norm - dist_outer_norm
        tensor = torch.from_numpy(dist_norm).type(torch.float32).unsqueeze(0)
        torch.save(tensor, save_dir.joinpath(label_path.stem + '.pt'))
        # 将dist_norm转为灰度图保存
        # dist_norm = (dist_norm + 1) / 2 * 255
        # cv2.imwrite(str(vis_dir.joinpath(label_path.stem + '>R.png')), dist_norm)
        # break


def kmeans_cluster(dataset_dir, labeled_ratio=0.1, k=3):
    '''
    数据集预处理：用全局ASM无约束计算各个样本的坐标，用于聚类
    '''
    # global_model = ActiveShapeModel(dataset_dir=dataset_dir, proj_name=proj_name, init_model=init_model, indexes='all')
    # global_model.get_trainset_b(max_bs=False)     # 如果已经存在，这行可以注释掉
    from models.ASM import ActiveShapeModel_torch
    from exps.configs.CAMUS.unet import Config

    proj_dir = Path(dataset_dir).joinpath('SSL')
    masm_path = proj_dir.joinpath('MASM', str(labeled_ratio))
    b_path = proj_dir.joinpath('SASM', str(labeled_ratio), 'check_label', 'bs', 'b.csv')
    coordinates = pd.read_csv(b_path, index_col=0)
    save_dir = masm_path.joinpath('{}_clusters'.format(k))
    # coordinates.set_index(0, inplace=True)
    eig_vals = torch.load(proj_dir.joinpath('SASM', str(labeled_ratio), 'model dir', 'eig_vals'))  # [10]
    weight = eig_vals / torch.sum(eig_vals)
    weighted_coordinate = coordinates * weight
    kmeans = KMeans(k, random_state=2).fit(weighted_coordinate)
    config = Config()
    # kmeans聚类，分开样本
    for cls in range(k):
        samples = {}
        samples['name'] = []
        cls_save_dir = save_dir.joinpath(str(cls))
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
        config.asm_save_dir = cls_save_dir
        asm_model = ActiveShapeModel_torch(config)
        # asm_model.get_trainset_b(max_bs=True)


def make_bbox_params(dataset_dir):
    '''
    制作shape bbox的参数：中心点坐标，宽、高、倾斜角度
    '''
    dataset_dir = Path(dataset_dir)
    label_dir = dataset_dir.joinpath('label_544x736')
    vis_dir = dataset_dir.joinpath('label_bbox_544x736')
    if not vis_dir.exists():
        vis_dir.mkdir(parents=True)
    results = {}
    for label_path in label_dir.iterdir():
        label = cv2.imread(str(label_path), -1)
        countors, _ = cv2.findContours(label, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(countors[0])
        results[label_path.stem] = [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]  # 中心(x,y), (宽，高), 旋转角度
        # for vis
        bbox = cv2.boxPoints(rect)
        vis = cv2.drawContours(label, [np.int0(bbox)], 0, 255, 2)
        cv2.imwrite(str(vis_dir.joinpath(label_path.name)), vis)
        # break
    with open(dataset_dir.joinpath('shape_bbox_544x736.json'), 'w') as f:
        f.write(json.dumps(results, cls=NpEncoder))


def SSL_split_trainset(root):
    '''
    用于SSL，将trainset切分出10%、20%，保存其名单
    '''
    root = Path(root)
    save_dir = root.joinpath('SSL', 'split')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    image_dir = root.joinpath('image_544x736')
    lst = sorted([x.name for x in image_dir.iterdir()])
    trainset = lst[: int(0.7 * len(lst))]
    random.shuffle(trainset)
    # testset = lst[int(0.7 * len(lst)) :]
    # test = {'test': testset}
    # with open(save_dir.joinpath('test.json'), 'w') as f:
    #     f.write(json.dumps(test))
    # ratios = [0.05, 0.1, 0.2, 1]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    for ratio in ratios:
        labeled_paths = trainset[: int(ratio * len(trainset))]
        unlabeled_paths = trainset[int(ratio * len(trainset)) :]
        d = {'labeled': labeled_paths, 'unlabeled': unlabeled_paths}
        with open(save_dir.joinpath(str(ratio) + '.json'), 'w') as f:
            f.write(json.dumps(d))


if __name__ == '__main__':
    # root = 'dataset/CAMUS/data/training'
    # dest = 'dataset/CAMUS/data/'

    # copy_to_dataset(root, dest)
    # get_landmarks(root, dest)
    # resize_dataset_shapes(dest)
    # make_bbox_params(dest)

    # # CAMUS test set: 540;
    # # ks = [int((i+1)*0.1*540) for i in range(10)]
    # # ps = [i+1 for i in range(10)]
    # from tqdm import tqdm
    # ks = [i * 10 for i in range(1, 55)]
    # for k in ks:
    #     kmeans_cluster(dest, k)
    # ks = [i for i in range(2, 16)]
    # ks = [i for i in range(16, 30)]
    # ks = [i for i in range(30, 100)]
    # for k in ks:
    #     kmeans_cluster(dest, k)

    root = 'dataset/CAMUS/data'
    # SSL_split_trainset(root)
    make_SDF(root)
    # lk = {0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5, 0.6: 7, 0.7: 8}
    # for l, k in lk.items():
    #     kmeans_cluster(root, l, k)
    # for k in range(1, 21):
    #     kmeans_cluster(root, 1, k)
