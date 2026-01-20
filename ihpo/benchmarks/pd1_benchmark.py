from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import PD1SearchSpace
from ..consts import PD1_TASKS
import numpy as np
import os
import pickle
from ConfigSpace import ConfigurationSpace
from carps.objective_functions.mfpbench import MFPBenchObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

class PD1Benchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/pd1/surrogates/') -> None:
        super().__init__()
        self._search_space = PD1SearchSpace(task)
        self._file_path = save_dir
        self.task = task
        self.surrogate_cache = {}
        self._setup_carps()

    def _setup_carps(self):
        """
            Load the trained surrogates representing the objective function.
        """
        self.objective_function = MFPBenchObjectiveFunction(
            benchmark_name="pd1",
            benchmark=self.task,
            metric = ["train_cost","valid_error_rate"]
        )
        self.configuration_space: ConfigurationSpace = self.objective_function.configspace
        print()
        

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        reformatted_cfg = self._reformat_cfg(cfg)
        trial_info = TrialInfo(
            config=reformatted_cfg,
            budget=budget
        )
        result = self.objective_function.evaluate(trial_info=trial_info)

        # round perforamnces, return this in the form of a BenchQueryResult

        benchmark_args = {
            'train_performance': - result.cost[0],
            'val_performance': - result.cost[1],
            'test_performance': - result.cost[1]
        }

        return BenchQueryResult(**benchmark_args)
                
        
    def _reformat_cfg(self, cfg: Dict) -> Dict:
        return {
            "lr_decay_factor" : cfg["hps.lr_hparams.decay_steps_factor"],
            "lr_initial" : cfg["hps.lr_hparams.initial_value"],
            "lr_power" : cfg["hps.lr_hparams.power"],
            "opt_momentum" : cfg["hps.opt_hparams.momentum"],
        }

    def get_min_and_max(self):
        return [0, 1]
    
    @property
    def search_space(self):
        return self._search_space