import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class Dataset0(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: ['train', 'labeled' or 'unlabeled'], ['test']
        '''
        self.config = config
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        self.label_dir = self.root.joinpath('label_544x736')
        self.mode = mode
        self.imgs = sorted([x for x in self.img_dir.iterdir()])
        if self.mode in ['train']:
            self.imgPath = self.imgs[: int(0.7 * len(self.imgs))]
        else:
            self.imgPath = self.imgs[int(0.7 * len(self.imgs)) :]

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_path = self.label_dir.joinpath(img_path.name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        return img, label, img_path.name

    def __len__(self):
        return len(self.imgPath)


class DataLoader0:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='train', batch_size=4):
        shuffle = True if mode == 'train' else False
        dataset = Dataset0(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


# ---------------------------------------------------------------------------------------------------------


class Dataset1_CAMUS(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        if mode == 'train_all':
            self.label_dir = config.save_path.joinpath('psudo_label', 'label_544x736')
            self.heatmap_dir = config.save_path.joinpath('psudo_label', 'heatmap', config.heatmap_dir_name)
            self.shapes = json.load(open(config.save_path.joinpath('psudo_label', 'shape_544x736.json')))
        else:
            self.label_dir = self.root.joinpath('label_544x736')
            self.heatmap_dir = self.root.joinpath('heatmap', config.heatmap_dir_name)
            self.shapes = json.load(open(self.root.joinpath('shape_544x736.json')))
        split = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_imgPath = [self.img_dir.joinpath(x) for x in split['labeled']]
        unlabeled_imgPath = [self.img_dir.joinpath(x) for x in split['unlabeled']]
        testset = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))
        testset_imgPath = [self.img_dir.joinpath(x) for x in testset['test']]
        if mode == 'train_labeled':
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('trainset_b.csv'), index_col=0)
            self.imgPath = labeled_imgPath
        elif mode == 'gen_unlabeled':
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('trainset_b.csv'), index_col=0)  # 实际上不会用到
            self.imgPath = unlabeled_imgPath
        elif mode == 'train_all':
            self.bs = pd.read_csv(Path(config.asm_plus_dir).joinpath('trainset_b.csv'), index_col=0)
            psudo_shapes = json.load(open(config.save_path.joinpath('gen_unlabeled', 'selected_psudo_shapes.json')))
            psudo_imgPath = [self.img_dir.joinpath(x + '.png') for x in psudo_shapes.keys()]
            self.imgPath = labeled_imgPath + psudo_imgPath
        else:
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('testset_b.csv'), index_col=0)
            self.imgPath = testset_imgPath

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_path = self.label_dir.joinpath(img_path.name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        heatmap = torch.load(self.heatmap_dir.joinpath(img_path.stem + '.pt'))
        heatmap = transforms.Resize((self.config.input_size[1], self.config.input_size[0]), antialias=True)(heatmap)
        shape = torch.Tensor(self.shapes[img_path.stem])  # / torch.Tensor(self.config.input_size)
        shape = torch.div(shape, self.config.resize_ratio, rounding_mode='trunc')
        b = torch.Tensor(1) if self.mode == 'gen_unlabeled' else torch.Tensor(self.bs.loc[img_path.stem])

        return img, label, heatmap, shape, b, img_path.name

    def __len__(self):
        return len(self.imgPath)


class Dataset1_ms(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_960x832')
        if mode == 'train_all':
            self.label_dir = config.save_path.joinpath('psudo_label', 'label_single_960x832')
            self.heatmap_dir = config.save_path.joinpath('psudo_label', 'heatmap', config.heatmap_dir_name)
            self.shapes = json.load(open(config.save_path.joinpath('psudo_label', 'shape_960x832.json')))
        else:
            self.label_dir = self.root.joinpath('label_single_960x832')
            self.heatmap_dir = self.root.joinpath('heatmap', config.heatmap_dir_name)
            self.shapes = json.load(open(self.root.joinpath('shape_960x832.json')))
        split = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_imgPath = [self.img_dir.joinpath(x) for x in split['labeled']]
        unlabeled_imgPath = [self.img_dir.joinpath(x) for x in split['unlabeled']]
        testset = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))
        testset_imgPath = [self.img_dir.joinpath(x) for x in testset['test']]
        if mode == 'train_labeled':
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('trainset_b.csv'), index_col=0)
            self.imgPath = labeled_imgPath
        elif mode == 'gen_unlabeled':
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('trainset_b.csv'), index_col=0)  # 实际上不会用到
            self.imgPath = unlabeled_imgPath
        elif mode == 'train_all':
            self.bs = pd.read_csv(Path(config.asm_plus_dir).joinpath('trainset_b.csv'), index_col=0)
            psudo_shapes = json.load(open(config.save_path.joinpath('gen_unlabeled', 'selected_psudo_shapes.json')))
            psudo_imgPath = [self.img_dir.joinpath(x + '.png') for x in psudo_shapes.keys()]
            self.imgPath = labeled_imgPath + psudo_imgPath
        else:
            self.bs = pd.read_csv(Path(config.asm_dir).joinpath('testset_b.csv'), index_col=0)
            self.imgPath = testset_imgPath

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_L_path = self.label_dir.joinpath(img_path.stem + '>L.png')
        label_L = Image.open(label_L_path)
        label_L = transforms.ToTensor()(label_L)
        label_L = torch.where(label_L > 0, torch.tensor(1.0), torch.tensor(0.0))
        label_R_path = self.label_dir.joinpath(img_path.stem + '>R.png')
        label_R = Image.open(label_R_path)
        label_R = transforms.ToTensor()(label_R)
        label_R = torch.where(label_R > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = torch.cat([label_L, label_R], dim=0)

        heatmap = torch.load(self.heatmap_dir.joinpath(img_path.stem + '.pt'))
        heatmap = transforms.Resize((self.config.input_size[1], self.config.input_size[0]), antialias=True)(heatmap)

        shape_L = torch.Tensor(self.shapes[img_path.stem]['L'])  # / torch.Tensor(self.config.input_size)
        shape_L = torch.div(shape_L, self.config.resize_ratio, rounding_mode='trunc')
        shape_R = torch.Tensor(self.shapes[img_path.stem]['R'])  # / torch.Tensor(self.config.input_size)
        shape_R = torch.div(shape_R, self.config.resize_ratio, rounding_mode='trunc')
        shape = torch.cat([shape_L, shape_R], dim=0)

        if self.mode == 'gen_unlabeled':
            b = torch.Tensor(1)
        else:
            b_L = torch.Tensor(self.bs.loc[img_path.stem + '>L'])
            b_R = torch.Tensor(self.bs.loc[img_path.stem + '>R'])
            b = torch.cat([b_L, b_R], dim=0)

        return img, label, heatmap, shape, b, img_path.name

    def __len__(self):
        return len(self.imgPath)


class DataLoader1:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='train', batch_size=4):
        shuffle = True if 'train' in mode else False
        dataset = Dataset1_ms(self.config, mode) if self.config.dataset_name == 'ms' else Dataset1_CAMUS(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


# ---------------------------------------------------------------------------------------------------------


class Dataset2(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: [0]; train or test, [1] stem or DSM
        '''
        self.config = config
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        self.label_dir = self.root.joinpath('label_544x736')
        self.heatmap_dir = self.root.joinpath('heatmap', config.heatmap_dir_name)
        self.shapes = json.load(open(self.root.joinpath('shape_544x736.json')))
        self.bs = json.load(open(Path(config.asm_save_dir).joinpath('b.json')))
        # self.bs_gt =
        self.mode = mode
        self.imgs = sorted([x for x in self.img_dir.iterdir()])
        if self.mode[0] in ['train']:
            self.imgPath = self.imgs[: int(0.7 * len(self.imgs))]
        else:
            self.imgPath = self.imgs[int(0.7 * len(self.imgs)) :]

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_path = self.label_dir.joinpath(img_path.name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        # quarter_size = (self.config.input_size[1] // 4, self.config.input_size[0] // 4)
        heatmap = torch.load(self.heatmap_dir.joinpath(img_path.stem + '.pt'))
        shape = torch.Tensor(self.shapes[img_path.stem])  # / torch.Tensor(self.config.input_size)
        b = torch.Tensor(self.bs[img_path.stem])

        return img, label, heatmap, shape, b, img_path.name

    def __len__(self):
        return len(self.imgPath)


class DataLoader2:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='train', batch_size=4):
        shuffle = True if mode == 'train' else False
        dataset = Dataset2(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


# ---------------------------------------------------------------------------------------------------------


class Dataset_coord(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: 'train' or'test'
        '''
        self.config = config
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        self.shapes = json.load(open(self.root.joinpath('shape_544x736.json')))
        # self.label_dir = self.root.joinpath('label_544x736')
        # self.sdf_dir = self.root.joinpath('SDF_tensor')
        # self.mode = mode
        self.imgs = sorted([x for x in self.img_dir.iterdir()])
        if mode == 'train':
            self.imgPath = self.imgs[: int(0.7 * len(self.imgs))]
        else:
            self.imgPath = self.imgs[int(0.7 * len(self.imgs)) :]

    def __getitem__(self, index):
        img_path = self.imgPath[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        pts = np.array(self.shapes[img_path.stem])[self.config.pt_idxes]
        pts = pts / np.array(self.config.input_size)
        pts = torch.from_numpy(pts).float()

        return img, pts, img_path.name

    def __len__(self):
        return len(self.imgPath)


class DataLoader_coord:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='train', batch_size=4):
        shuffle = True if mode == 'train' else False
        dataset = Dataset_coord(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


# ---------------------------------------------------------------------------------------------------------


class Dataset_MT_CAMUS(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        # self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        self.label_dir = self.root.joinpath('label_544x736')
        trainset_names = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_names = trainset_names['labeled']
        unlabeled_names = trainset_names['unlabeled']
        testset_names = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))['test']
        if mode == 'labeled':
            self.imgNames = labeled_names
        elif mode == 'unlabeled':
            self.imgNames = unlabeled_names
        elif mode == 'test':
            self.imgNames = testset_names
        else:
            raise ValueError('mode should be labeled or unlabeled or test')

    def __getitem__(self, index):
        img_name = self.imgNames[index]
        img_path = self.img_dir.joinpath(img_name)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_path = self.label_dir.joinpath(img_name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        return img, label, img_name

    def __len__(self):
        return len(self.imgNames)


class Dataset_MT_ms(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        # self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_960x832')
        self.label_dir = self.root.joinpath('label_single_960x832')
        trainset_names = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_names = trainset_names['labeled']
        unlabeled_names = trainset_names['unlabeled']
        testset_names = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))['test']
        if mode == 'labeled':
            self.imgNames = labeled_names
        elif mode == 'unlabeled':
            self.imgNames = unlabeled_names
        elif mode == 'test':
            self.imgNames = testset_names
        else:
            raise ValueError('mode should be labeled or unlabeled or test')

    def __getitem__(self, index):
        img_name = self.imgNames[index]
        img_path = self.img_dir.joinpath(img_name)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_L_path = self.label_dir.joinpath(img_path.stem + '>L.png')
        label_L = Image.open(label_L_path)
        label_L = transforms.ToTensor()(label_L)
        label_L = torch.where(label_L > 0, torch.tensor(1.0), torch.tensor(0.0))
        label_R_path = self.label_dir.joinpath(img_path.stem + '>R.png')
        label_R = Image.open(label_R_path)
        label_R = transforms.ToTensor()(label_R)
        label_R = torch.where(label_R > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = torch.cat([label_L, label_R], dim=0)

        return img, label, img_name

    def __len__(self):
        return len(self.imgNames)


class DataLoader_MT:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='labeled', batch_size=4):
        dataset = Dataset_MT_CAMUS(self.config, mode) if self.config.dataset_name == 'CAMUS' else Dataset_MT_ms(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


# ---------------------------------------------------------------------------------------------------------


class Dataset_DTC_CAMUS(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        # self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_544x736')
        self.label_dir = self.root.joinpath('label_544x736')
        self.sdf_dir = self.root.joinpath('SDF_tensor', 'distance')
        trainset_names = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_names = trainset_names['labeled']
        unlabeled_names = trainset_names['unlabeled']
        testset_names = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))['test']
        if mode == 'labeled':
            self.imgNames = labeled_names
        elif mode == 'unlabeled':
            self.imgNames = unlabeled_names
        elif mode == 'test':
            self.imgNames = testset_names
        else:
            raise ValueError('mode should be labeled or unlabeled or test')

    def __getitem__(self, index):
        img_name = self.imgNames[index]
        img_path = self.img_dir.joinpath(img_name)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_path = self.label_dir.joinpath(img_name)
        label = Image.open(label_path)
        label = transforms.ToTensor()(label)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))

        sdf = torch.load(self.sdf_dir.joinpath(img_path.stem + '.pt'))
        sdf = transforms.Resize((self.config.input_size[1], self.config.input_size[0]), antialias=True)(sdf)

        return img, label, sdf, img_name

    def __len__(self):
        return len(self.imgNames)


class Dataset_DTC_ms(torch.utils.data.Dataset):
    def __init__(self, config, mode) -> None:
        '''
        mode: train_labeled or train_all or gen_unlabeled
        '''
        self.config = config
        # self.mode = mode
        self.root = Path(config.dataset_path)
        self.img_dir = self.root.joinpath('image_960x832')
        self.label_dir = self.root.joinpath('label_single_960x832')
        self.sdf_dir = self.root.joinpath('SDF_tensor', 'distance')
        trainset_names = json.load(open(self.root.joinpath('SSL', 'split', '{}.json'.format(self.config.labeled_ratio))))
        labeled_names = trainset_names['labeled']
        unlabeled_names = trainset_names['unlabeled']
        testset_names = json.load(open(self.root.joinpath('SSL', 'split', 'test.json')))['test']
        if mode == 'labeled':
            self.imgNames = labeled_names
        elif mode == 'unlabeled':
            self.imgNames = unlabeled_names
        elif mode == 'test':
            self.imgNames = testset_names
        else:
            raise ValueError('mode should be labeled or unlabeled or test')

    def __getitem__(self, index):
        img_name = self.imgNames[index]
        img_path = self.img_dir.joinpath(img_name)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        label_L_path = self.label_dir.joinpath(img_path.stem + '>L.png')
        label_L = Image.open(label_L_path)
        label_L = transforms.ToTensor()(label_L)
        label_L = torch.where(label_L > 0, torch.tensor(1.0), torch.tensor(0.0))
        label_R_path = self.label_dir.joinpath(img_path.stem + '>R.png')
        label_R = Image.open(label_R_path)
        label_R = transforms.ToTensor()(label_R)
        label_R = torch.where(label_R > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = torch.cat([label_L, label_R], dim=0)

        sdf = torch.load(self.sdf_dir.joinpath(img_path.stem + '.pt'))
        sdf = transforms.Resize((self.config.input_size[1], self.config.input_size[0]), antialias=True)(sdf)

        return img, label, sdf, img_name

    def __len__(self):
        return len(self.imgNames)


class DataLoader_DTC:
    def __init__(self, config) -> None:
        self.config = config

    def load_data(self, mode='labeled', batch_size=4):
        dataset = Dataset_DTC_CAMUS(self.config, mode) if self.config.dataset_name == 'CAMUS' else Dataset_DTC_ms(self.config, mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return loader


if __name__ == '__main__':
    # root = '/hdd/data/dataset/casia2/AngleHD-SO_muscle'
    # root = 'dataset/casia2/AngleHD-SO_muscle_2nd'
    # dataset = Dataset(root, 'infer_nolabel', (1056, 1440))
    # img = dataset[0]
    # from exps.configs.unet.Ciliary_id4 import Config
    # from ..exps.configs.unet.Ciliary_id4 import Config
    # config = Config()
    # loader = DataLoader_patch(config).load_data('train', 0, 1)
    # # print(len(loader))
    # for img, label in loader:
    pass
