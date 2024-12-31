import os
import logging
from framework.package_tools.path_utils import path_to_package

logger = logging.getLogger()


class BasicConfig:
    """
    所有的新实验设置都必须创建一个新的`BasicConfig`实例，然后进行增删。
    """

    save_path: str
    script: str
    gpu: str

    def __init__(self) -> None:
        # self.project = 'ciliary'
        self.gpu = '0'
        self.train_batch_size = 4
        # self.save_path = 'save'
        self.epochs = 1000
        self.lr = 1e-4
        self.num_workers = 4
        self.input_size = None
        self.val_batch_size = 1
        self.script = ''
        self.path_to_init = []
        # self.evaluate = False

    def build(self):
        self.init_path()
        self.path_to_package()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

    def log(self):
        logger.info('------experiment overview------')

    def path_to_package(self):
        if '/' in self.script or self.script.endswith('.py'):
            self.script = path_to_package(self.script)

    def init_path(self):
        for path in self.path_to_init:
            if not path.exists():
                path.mkdir(parents=True)

    def register_path(self, paths):
        # self.path_to_init.append(*paths)
        self.path_to_init += paths
