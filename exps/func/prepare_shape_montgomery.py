import cv2
import math
import json
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
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

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

def resize_copy_to_dataset():
    root = Path('/home/yb/4t/data/public_dataset/MontgomerySet')
    dest = Path('dataset/Montgomery')
    dest.mkdir(parents=True, exist_ok=True)

    root_image_folder = root.joinpath('CXR_png')
    root_label_folder = root.joinpath('ManualMask')

    dest_image_folder = dest.joinpath('image')
    dest_label_folder = dest.joinpath('label')
    dest_image_folder.mkdir(parents=True, exist_ok=True)
    dest_label_folder.mkdir(parents=True, exist_ok=True)

    # # 遍历源图片文件夹中的所有图片
    # for image_path in root_image_folder.iterdir():
    #     if image_path.name.startswith('.'):
    #         continue
    #     print(image_path.name)
    #     image = cv2.imread(str(image_path))
    #     image_resized = cv2.resize(image, (1024, 1024))
    #     save_path = dest_image_folder.joinpath(image_path.name)
    #     cv2.imwrite(str(save_path), image_resized)
    #     # break

    # 处理左右肺标签
    left_mask_folder = root_label_folder.joinpath('leftMask')
    right_mask_folder = root_label_folder.joinpath('rightMask')
    
    # 处理左肺标签
    for mask_path in left_mask_folder.iterdir():
        if mask_path.name.startswith('.'):
            continue
        mask = cv2.imread(str(mask_path))
        mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        save_name = mask_path.stem + '_L.png'
        save_path = dest_label_folder.joinpath(save_name)
        cv2.imwrite(str(save_path), mask_resized)
        # break
    
    # 处理右肺标签
    for mask_path in right_mask_folder.iterdir():
        if mask_path.name.startswith('.'):
            continue
        mask = cv2.imread(str(mask_path))
        # 最近邻resize
        mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        save_name = mask_path.stem + '_R.png'
        save_path = dest_label_folder.joinpath(save_name)
        cv2.imwrite(str(save_path), mask_resized)
        # break

def sample_points(points, start_idx, end_idx, num_points):
    """
    在边缘点序列上均匀采样点
    """
    # 提取需要采样的边缘点段
    if end_idx < start_idx:
        segment = np.vstack((points[start_idx:], points[:end_idx+1]))
    else:
        segment = points[start_idx:end_idx+1]
    
    # 计算总路径长度
    total_length = 0
    for i in range(len(segment)-1):
        total_length += np.sqrt(np.sum((segment[i+1] - segment[i])**2))
    
    # 计算采样间隔
    interval = total_length / (num_points - 1)
    
    sampled_points = []
    sampled_points.append(segment[0])  # 起点
    
    current_point = segment[0]
    current_idx = 0
    accumulated_length = 0
    
    for i in range(1, num_points-1):
        target_length = i * interval
        
        # 找到目标长度所在的线段
        while current_idx < len(segment)-1:
            next_point = segment[current_idx + 1]
            segment_length = np.sqrt(np.sum((next_point - segment[current_idx])**2))
            
            if accumulated_length + segment_length >= target_length:
                # 计算在当前线段上的插值位置
                remaining = target_length - accumulated_length
                ratio = remaining / segment_length
                
                # 计算方向向量
                direction = next_point - segment[current_idx]
                
                # 计算采样点
                point = segment[current_idx] + ratio * direction
                point = np.round(point).astype(np.int32)
                point = np.clip(point, 0, 1024)
                sampled_points.append(point)
                break
            
            accumulated_length += segment_length
            current_idx += 1
    
    sampled_points.append(segment[-1])  # 终点
    return np.array(sampled_points)

def get_landmarks():
    '''
    1. 读取labelme的json文件，获取3个key landmarks (up, inner, outer)
    2. 根据三个关键点在肺边缘上分段采样：
       - inner到up: 31点
       - up到outer: 31点
       - outer到inner: 11点
    3. 将结果按顺序保存：inner -> inner-up -> up -> up-outer -> outer -> outer-inner
    4. �����别保存左右肺的点集
    '''
    root = Path('/home/yb/4t/data/labeled/Montgomery/label')
    dest = Path('dataset/Montgomery')
    dest.mkdir(parents=True, exist_ok=True)
    ignore_list = ['MCUCXR_0043_0', 'MCUCXR_0058_0']
    total = {}
    
    for file_path in root.glob('*.json'):
        if file_path.name.startswith('.'):
            continue
        if file_path.stem[:-2] in ignore_list:
            continue
        print(file_path.name)
        
        # 获取基础名称和左右标识
        base_name = file_path.stem[:-2]  # 去掉_L或_R
        side = file_path.stem[-1]  # L或R
        
        # 如果base_name不在total中，创建新条目
        if base_name not in total:
            total[base_name] = {"L": [], "R": []}
        
        data = json.load(open(file_path))
        
        # 获取三个关键点坐标
        key_points = {}
        for shape in data['shapes']:
            point = np.array(shape['points'][0])
            point = np.round(point).astype(np.int32)
            point = np.clip(point, 0, 1024)
            key_points[shape['label']] = point
        
        up = key_points['up']
        inner = key_points['inner'] 
        outer = key_points['outer']
        
        # 读取对应的label图像
        label_name = file_path.stem + '.png'
        label_path = dest.joinpath('label', label_name)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        # 获取边缘点
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge_points = contours[0].squeeze()
        
        # 找到最接近三个关键点的边缘点索引
        up_idx = np.argmin(np.sum((edge_points - up) ** 2, axis=1))
        inner_idx = np.argmin(np.sum((edge_points - inner) ** 2, axis=1))
        outer_idx = np.argmin(np.sum((edge_points - outer) ** 2, axis=1))
        
        # 确保点的顺序是顺时针或逆时针
        # 通过判断inner到up的方向来确定是否需要反转点序
        if abs(up_idx - inner_idx) > len(edge_points) // 2:
            # 如果两点索引差距太大，说明跨越了数组边界
            if up_idx > inner_idx:
                inner_idx += len(edge_points)
            else:
                up_idx += len(edge_points)
        
        # 根据索引的大小关系确定是否需要反转点序
        if up_idx < inner_idx:
            edge_points = np.flip(edge_points, axis=0)
            # 重新计算索引
            up_idx = len(edge_points) - 1 - up_idx
            inner_idx = len(edge_points) - 1 - inner_idx
            outer_idx = len(edge_points) - 1 - outer_idx
        
        # 使用边缘点替代原始关键点
        up = edge_points[up_idx % len(edge_points)]
        inner = edge_points[inner_idx % len(edge_points)]
        outer = edge_points[outer_idx % len(edge_points)]
        
        # 定义采样点数
        num_points_up_inner = 31
        num_points_up_outer = 31
        num_points_inner_outer = 11
        
        # 在三段上分别采样
        inner_up_points = sample_points(edge_points, inner_idx % len(edge_points), up_idx % len(edge_points), num_points_up_inner)
        up_outer_points = sample_points(edge_points, up_idx % len(edge_points), outer_idx % len(edge_points), num_points_up_outer)
        outer_inner_points = sample_points(edge_points, outer_idx % len(edge_points), inner_idx % len(edge_points), num_points_inner_outer)
        
        # 按顺序组织所有点
        all_points = []
        # 1. inner点和inner到up的点（包含inner，不包含up）
        all_points.extend(inner_up_points[:-1].tolist())
        # 2. up点和up到outer的点（包含up，不包含outer）
        all_points.extend(up_outer_points[:-1].tolist())
        # 3. outer点和outer到inner的点（包含outer，不包含inner）
        all_points.extend(outer_inner_points[:-1].tolist())
        
        # 保存结果到对应的左右肺
        total[base_name][side] = all_points
    
    # 保存到json文件
    save_path = dest.joinpath('shape.json')
    with open(save_path, 'w') as f:
        json.dump(total, f, cls=NpEncoder)

def vis_landmark():
    '''
    读取shape.json和label，landmarks用线连起来，画在label上
    所有点都标注序号，其中关键点(0,30,60)用红色显示，其他点用绿色显示，连线用黄色
    '''
    root = Path('dataset/Montgomery')
    label_dir = root.joinpath('label_single')
    save_dir = root.joinpath('vis_landmarks')
    save_dir.mkdir(exist_ok=True)
    
    # 读取shape.json
    with open(root.joinpath('shape.json'), 'r') as f:
        landmarks = json.load(f)
    
    # 关键点索引
    key_indices = [0, 30, 60]  # inner, up, outer
    key_names = ['inner(0)', 'up(30)', 'outer(60)']  # 关键点标注文字
    
    # 遍历每张图像
    for name, points in landmarks.items():
        # 读取label图像
        label_path = label_dir.joinpath(name + '.png')
        label = cv2.imread(str(label_path))
        
        # 将landmarks转换为numpy数组
        points = np.array(points)
        
        # 画点和标注序号
        for i, pt in enumerate(points):
            if i in key_indices:
                # 关键点用红色，大一点
                cv2.circle(label, (pt[0], pt[1]), 3, (0, 0, 255), -1)
                # 添加序号标注
                idx = key_indices.index(i)
                text = key_names[idx]
                color = (0, 0, 255)  # 红色
            else:
                # 其他点用绿色，小一点
                cv2.circle(label, (pt[0], pt[1]), 2, (0, 255, 0), -1)
                # 添加序号标注
                text = str(i)
                color = (0, 255, 0)  # 绿色
            
            # 标注序号
            cv2.putText(label, 
                      text, 
                      (pt[0]+5, pt[1]-5),  # 文字位置略微偏移
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.4,  # 字体大小稍微调小
                      color,  # 与点的颜色相同
                      1)  # 线宽
        
        # 画线（黄色）
        for i in range(len(points)-1):
            cv2.line(label, 
                    (points[i][0], points[i][1]), 
                    (points[i+1][0], points[i+1][1]), 
                    (0, 255, 255),  # 黄色线
                    1)
        # 连接最后一个点和第一个点，成闭合轮廓
        cv2.line(label,
                (points[-1][0], points[-1][1]),
                (points[0][0], points[0][1]),
                (0, 255, 255),  # 黄色线
                1)
        
        # 保存结果
        save_path = save_dir.joinpath(name + '_landmarks.png')
        cv2.imwrite(str(save_path), label)


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


def make_heatmap():
    '''
    制作点的热力图
    '''
    root = Path('dataset/Montgomery')
    pt_indexes = list(range(70))
    WH = (1024, 1024)  # 修正高度值
    sigmma = 3
    gamma = 7

    save_path = root.joinpath('heatmap')
    vis_path = save_path.joinpath('vis_' + str(len(pt_indexes) * 2) + '_' + str(sigmma))
    vis_path.mkdir(parents=True, exist_ok=True)
    tensor_path = save_path.joinpath('tensor_' + str(len(pt_indexes) * 2) + '_' + str(sigmma))
    tensor_path.mkdir(parents=True, exist_ok=True)
    shapes = json.load(open(root.joinpath('shape.json')))

    def gen_heatmap(shape, pt_indexes, WH, sigmma, gamma):
        masks = np.zeros((len(pt_indexes), WH[1], WH[0]), dtype=np.float32)
        for i in range(len(pt_indexes)):
            pt_index = pt_indexes[i]
            mask = gaussian(shape[pt_index], WH[1], WH[0], sigmma, gamma)
            masks[i] = mask
        for i in range(masks.shape[0]):
            factor = 1 / np.max(masks[i])
            masks[i] *= factor
        masks_tensor = torch.from_numpy(masks).type(torch.float32)
        return masks_tensor

    for name, couple in shapes.items():
        shape_L_mask_tensor = gen_heatmap(couple['L'], pt_indexes, WH, sigmma, gamma)
        shape_R_mask_tensor = gen_heatmap(couple['R'], pt_indexes, WH, sigmma, gamma)
        shape_mask_tensor = torch.cat((shape_L_mask_tensor, shape_R_mask_tensor), dim=0)
        torch.save(shape_mask_tensor, tensor_path.joinpath(name + '.pt'))
        
        # # 可视化每个通道的热力图
        # for i in range(shape_mask_tensor.shape[0]):
        #     # 获取当前通道
        #     heatmap = shape_mask_tensor[i].numpy()
        #     # 转换为0-255范围的图像
        #     heatmap = (heatmap * 255).astype(np.uint8)
        #     # 应用伪彩色映射
        #     # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     # 保存图像
        #     side = 'L' if i < len(pt_indexes) else 'R'
        #     point_idx = i % len(pt_indexes)
        #     save_name = f"{name}_{side}_point{point_idx}.png"
        #     cv2.imwrite(str(vis_path.joinpath(save_name)), heatmap)
        
        # break

def make_label_couple():
    '''
    遍历label_single，合并LR为同一张图像
    '''
    root = Path('dataset/Montgomery')
    label_single_dir = root.joinpath('label_single')
    dest_dir = root.joinpath('label')
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有标签文件
    label_files = sorted([x for x in label_single_dir.iterdir() if x.name.endswith('.png')])
    
    # 按基础名称分组
    base_names = set()
    for file in label_files:
        base_name = file.stem[:-2]  # 去掉_L或_R
        base_names.add(base_name)
    
    # 处理每对图像
    for base_name in base_names:
        # 读取左右肺标签
        label_L = cv2.imread(str(label_single_dir.joinpath(f"{base_name}_L.png")), cv2.IMREAD_GRAYSCALE)
        label_R = cv2.imread(str(label_single_dir.joinpath(f"{base_name}_R.png")), cv2.IMREAD_GRAYSCALE)
        
        # 合并左右肺
        label_combined = np.zeros_like(label_L)
        label_combined[label_L > 0] = 255
        label_combined[label_R > 0] = 255
        
        # 保存合并后的标签
        cv2.imwrite(str(dest_dir.joinpath(f"{base_name}.png")), label_combined)

def SSL_split_trainset():
    '''
    用于SSL，将trainset切分出10%、20%，保存其名单
    '''
    root = Path('dataset/Montgomery')
    save_dir = root.joinpath('SSL', 'split')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    image_dir = root.joinpath('image')
    lst = sorted([x.name for x in image_dir.iterdir()])
    trainset = lst[: int(0.7 * len(lst))]
    random.shuffle(trainset)
    testset = lst[int(0.7 * len(lst)) :]
    test = {'test': testset}
    with open(save_dir.joinpath('test.json'), 'w') as f:
        f.write(json.dumps(test))
    # ratios = [0.05, 0.1, 0.2, 1]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    for ratio in ratios:
        labeled_paths = trainset[: int(ratio * len(trainset))]
        unlabeled_paths = trainset[int(ratio * len(trainset)) :]
        d = {'labeled': labeled_paths, 'unlabeled': unlabeled_paths}
        with open(save_dir.joinpath(str(ratio) + '.json'), 'w') as f:
            f.write(json.dumps(d))

if __name__ == '__main__':
    # resize_copy_to_dataset()
    # get_landmarks()
    # vis_landmark()
    # make_heatmap()
    make_label_couple()
    # SSL_split_trainset()