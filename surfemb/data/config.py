from collections import defaultdict


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'

config['motor'] = motor = DatasetConfig()
motor.model_folder = 'models_cad'
motor.test_folder = 'test_primesense'
motor.train_folder = 'train_pbr'
motor.img_ext = 'jpg'
motor.depth_ext = 'png'

config['motor'] = motor = DatasetConfig()
motor.model_folder = 'models_cad'
motor.test_folder = 'test_primesense'
motor.train_folder = 'train_pbr'
motor.img_ext = 'jpg'
motor.depth_ext = 'png'

config['tless_mod'] = tless_mod = DatasetConfig()
tless_mod.model_folder = 'models_cad'
tless_mod.test_folder = 'test_primesense'
tless_mod.train_folder = 'train_pbr'
tless_mod.img_ext = 'jpg'
tless_mod.depth_ext = 'png'


config['tlessmod'] = tlessmod = DatasetConfig()
tlessmod.model_folder = 'models_cad'
tlessmod.test_folder = 'test_primesense'
tlessmod.train_folder = 'train_pbr'
tlessmod.img_ext = 'jpg'
tlessmod.depth_ext = 'png'