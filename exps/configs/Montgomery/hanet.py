import logging
from pathlib import Path
from framework.config.basic import BasicConfig
from dataloader.dataloader import DataLoader1
from models.DSM import *

logger = logging.getLogger()


class Config(BasicConfig):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = 'Montgomery'
        self.labeled_ratio = 0.3
        self.train_mode = 'train_labeled'  # train_labeled or train_all
        self.script = 'exps/scripts/ASM_Montgomery.py'
        # self.script = 'exps/scripts/train.py'

        self.dataloader = DataLoader1
        self.heatmap_dir_name = 'tensor_140_3'
        self.dataset_path = Path('dataset/Montgomery')
        self.save_path = Path('output/Montgomery/unet').joinpath(str(self.labeled_ratio))
        # self.save_path = Path('output/ms/unet').joinpath('0.1_sigmma_4,lr-3')
        self.infer_part = 'left'  # stem left finetune
        self.infer_name = 'bestDice'  # minLoss bestDice final
        self.resize_ratio = 2
        self.input_size = (1024 // self.resize_ratio, 1024 // self.resize_ratio)  # WH
        # self.stem_net = UNet1
        self.e4_net = Encoder_4layer
        self.d4_net = Decoder_4layer
        self.stem = UNet2_2
        self.decoder_conv = 'conv'  # conv or res_conv
        self.stem_out_ch = [2, 6]  # 3 for CAMUS, 2 for Montgomery
        self.channels = [16, 32, 64, 128, 256]
        self.pt_idxes = [0, 30, 60, 70, 100, 130]
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.epochs = 1000
        self.snapshot = 100
        self.step = 20
        self.gpu = '0'
        self.lr = 1e-4

        # for ASM
        self.asm_dir = self.dataset_path.joinpath('SSL', 'ASM', str(self.labeled_ratio), '1_clusters')  # for labeled data
        self.asm_plus_dir = self.save_path.joinpath('ASM', '1_clusters')
        # self.lk = {0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5, 0.5: 6, 0.6: 7, 0.7: 8, 1: 10}
        # self.k = self.lk[self.labeled_ratio]
        self.k = 10  # clusters number
        self.total_pts_num = 140
        self.asm_max_bs = False
        self.asm_vis_ssm = True

    def build(self):
        self.register_path([self.save_path, self.asm_dir, self.asm_plus_dir])
        super().build()
        logger.info('Save path has been registered.')
        self.log()

    def log(self):
        '''
        将Config配置写入log
        '''
        super().log()
        var = vars(self)
        for key in var:
            if 'net' in key:
                continue
            logger.info('{}:{}'.format(key, var[key]))


def get_config():
    """
    返回的配置必须是`framework.config.BasicConfig`的实例，函数名字必须为`get_config`。
    """
    return Config()
