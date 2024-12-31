import logging
from pathlib import Path
from framework.config.basic import BasicConfig
from dataloader.dataloader import DataLoader1
from models.DSM import *

logger = logging.getLogger()


class Config(BasicConfig):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = 'CAMUS'
        self.labeled_ratio = 0.3
        self.train_mode = 'train_labeled'  # train_labeled or train_all
        # self.script = 'exps/scripts/ASM_CAMUS.py'
        self.script = 'exps/scripts/train.py'

        self.dataloader = DataLoader1
        self.heatmap_dir_name = 'tensor_61_3'
        self.dataset_path = Path('dataset/CAMUS/data')
        # self.save_path = Path('output/CAMUS/unet').joinpath(str(self.labeled_ratio), 'loss_b')
        self.save_path = Path('output/CAMUS/unet').joinpath(str(self.labeled_ratio))
        self.infer_part = 'finetune'  # stem left finetune
        self.infer_name = 'bestDice'  # minLoss bestDice final
        self.resize_ratio = 2
        self.input_size = (544 // self.resize_ratio, 736 // self.resize_ratio)
        # self.input_size = (544, 736)
        # self.stem_net = UNet1
        self.e5_net = Encoder_5layer
        self.e4_net = Encoder_4layer
        self.e3_net = Encoder_3layer
        self.d5_net = Decoder_5layer
        self.d4_net = Decoder_4layer
        self.d3_net = Decoder_3layer
        self.mlp_net = MLP
        self.stem = UNet2_2
        self.decoder_conv = 'conv'  # conv or res_conv
        self.stem_out_ch = [1, 3]  # 3 for CAMUS, 4 for CM540
        self.channels = [16, 32, 64, 128, 256]
        self.pt_idxes = [0, 30, 60]
        self.train_batch_size = 8
        self.val_batch_size = 1
        self.epochs = 600
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
        self.total_pts_num = 61
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
