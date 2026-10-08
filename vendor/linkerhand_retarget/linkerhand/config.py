from .utils import *


class HandConfig():
    def __init__(self, robot_dir: str, config_path: str):
        package_share_dir = config_path
        self.handconfig = read_yaml(os.path.join(package_share_dir, 'config', 'hand_config.yml'))
        self.baseconfig = read_yaml(os.path.join(package_share_dir, 'config', 'base_config.yml'))
        self.retagetconfig = read_yaml(os.path.join(package_share_dir, 'config', 'retarget_config.yml'))
        self.modelconfig = read_yaml(os.path.join(package_share_dir, 'config', 'model_config.yml'))

        self.robot_dir = robot_dir

        self.bodypose = read_yaml(
            os.path.join(package_share_dir, 'config', f'{self.baseconfig["humanset"]["bodyfile"]}.yml'))
        self.targetpose = read_yaml(
            os.path.join(package_share_dir, 'config', f'{self.baseconfig["humanset"]["targethandfile"]}.yml'))
        self.retagetconfig = read_yaml(os.path.join(package_share_dir, 'config', 'retarget_config.yml'))

