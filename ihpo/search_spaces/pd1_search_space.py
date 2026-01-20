from .search_space import SearchSpace
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from syne_tune.config_space import Float, loguniform
from copy import deepcopy

class PD1SearchSpace(SearchSpace):

    def __init__(self, task) -> None:
        super().__init__()
        self.task = task
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x_sample = {}
            for name, val in self.search_space_definition.items():
                if not val['is_log']:
                    x = np.random.uniform(val['min'], val['max'])
                else:
                    x = np.exp(np.random.uniform(np.log(val['min']), np.log(val['max'])))
                x_sample[name] = x

            samples.append(x_sample)
        return samples
    
    def get_neighbors(self, config: dict, num_cont_neighbors=6):
        neighbors = []
        if config is None:
            return self.sample()
        for name, param_def in self.search_space_definition.items():
            curr_value = config[name]
            min_val = self.search_space_definition[name]['min']
            max_val = self.search_space_definition[name]['max']
            dist = max_val - min_val
            rnd_left = np.random.uniform(0, 0.2*dist, size=(num_cont_neighbors // 2))
            rnd_right = np.random.uniform(0, -0.2*dist, size=(num_cont_neighbors // 2))
            rnd = np.concatenate((rnd_left, rnd_right))
            rnd += curr_value
            ns = list(rnd)
            for neighbor in ns:
                curr_config = deepcopy(config)
                curr_config[name] = neighbor
                neighbors.append(curr_config)
        return neighbors
    
    def to_configspace(self):
        config_space = ConfigurationSpace()
        for key, val in self.search_space_definition.items():
            hp = UniformFloatHyperparameter(key, val['min'], val['max'], log=val['is_log'])
            config_space.add_hyperparameter(hp)
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.config_space.items():
            if val['is_log']:
                hp = loguniform(val['min'], val['max'])
            else:
                hp = Float(val['min'], val['max'])
            config_space[key] = hp
                    
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        for key, val in self.search_space_definition.items():
            hp = {
                    "parameter_type": "real",
                    "values": [val['min'], val['max']]
                }
            config_space[key] = hp
        return config_space
    
    def is_valid(self, config):
        return True
    
    def get_search_space_definition(self):
        if self.task == "cifar100_wideresnet_2048":
            search_space_definition = {
                'hps.lr_hparams.decay_steps_factor': {
                    'type': MetaType.REAL,
                    'min': 0.010093,
                    'max': 0.989012,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.lr_hparams.initial_value': {
                    'type': MetaType.REAL,
                    'min': 0.00001,
                    'max': 9.779176,
                    'dtype': 'float',
                    'is_log': True,
                },
                'hps.lr_hparams.power': {
                    'type': MetaType.REAL,
                    'min': 0.100708,
                    'max': 1.999376,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.opt_hparams.momentum': {
                    'type': MetaType.REAL,
                    'min': 0.000059, # according to PD1 paper this is 1e-3, but in data there are values starting at 1e-5
                    'max': 0.998993,
                    'dtype': 'float',
                    'is_log': True,
                }
            }
        elif self.task == "imagenet_resnet_512":
            search_space_definition = {
                'hps.lr_hparams.decay_steps_factor': {
                    'type': MetaType.REAL,
                    'min': 0.010294,
                    'max': 0.989753,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.lr_hparams.initial_value': {
                    'type': MetaType.REAL,
                    'min': 0.00001,
                    'max': 9.774312,
                    'dtype': 'float',
                    'is_log': True,
                },
                'hps.lr_hparams.power': {
                    'type': MetaType.REAL,
                    'min': 0.100225,
                    'max': 1.999326,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.opt_hparams.momentum': {
                    'type': MetaType.REAL,
                    'min': 5.9e-05, # according to PD1 paper this is 1e-3, but in data there are values starting at 1e-5
                    'max':  0.998993,
                    'dtype': 'float',
                    'is_log': True,
                }
            }
        elif self.task == "lm1b_transformer_2048":
            search_space_definition = {
                'hps.lr_hparams.decay_steps_factor': {
                    'type': MetaType.REAL,
                    'min': 0.010543,
                    'max': 0.9885653,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.lr_hparams.initial_value': {
                    'type': MetaType.REAL,
                    'min': 1e-05,
                    'max': 9.986256,
                    'dtype': 'float',
                    'is_log': True,
                },
                'hps.lr_hparams.power': {
                    'type': MetaType.REAL,
                    'min': 0.100811,
                    'max': 1.999659,
                    'dtype': 'float',
                    'is_log': False,
                },
               'hps.opt_hparams.momentum': {
                    'type': MetaType.REAL,
                    'min': 5.9e-05, # according to PD1 paper this is 1e-3, but in data there are values starting at 1e-5
                    'max':  0.9989986,
                    'dtype': 'float',
                    'is_log': True,
                }
            }
        elif self.task == "translatewmt_xformer_64":
            search_space_definition = {
                'hps.lr_hparams.decay_steps_factor': {
                    'type': MetaType.REAL,
                    'min': 0.0100221257,
                    'max': 0.988565263,
                    'dtype': 'float',
                    'is_log': False,
                },
                'hps.lr_hparams.initial_value': {
                    'type': MetaType.REAL,
                    'min': 1.00276e-05,
                    'max': 9.8422475735,
                    'dtype': 'float',
                    'is_log': True,
                },
                'hps.lr_hparams.power': {
                    'type': MetaType.REAL,
                    'min': 0.1004250993,
                    'max': 1.9985927056,
                    'dtype': 'float',
                    'is_log': False,
                },
               'hps.opt_hparams.momentum': {
                    'type': MetaType.REAL,
                    'min': 5.86114e-05, # according to PD1 paper this is 1e-3, but in data there are values starting at 1e-5
                    'max': 0.9989999746,
                    'dtype': 'float',
                    'is_log': True,
                }
            }
        return search_space_definition

    def change_search_space(self, key, value):
        self.search_space_definition[key] = value
