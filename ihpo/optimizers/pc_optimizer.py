from typing import Dict, List
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform
from ..utils.ibo import create_quantile_buckets, create_buckets, compute_bin_number
import numpy as np
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.Base import Context, Sum, assign_ids
from spn.structure.StatisticalTypes import MetaType
from numpy.random.mtrand import RandomState
from sklearn.metrics import r2_score
import scipy
import warnings
from time import time
import pandas as pd
import gower
from ConfigSpace import Configuration
from copy import deepcopy
# PRiors from DYNABO
PRIOR_ORIGIN_BASE_PATH = '/mnt/home/lfehring/DynaBO/benchmark_data/prior_data/mfpbench/cluster'

warnings.filterwarnings("ignore")

"""
    Implementation of the IBO-HPC optimizer.
    This optimizer fits a PC to the joint hyperparameter-evaluation space and can select configurations using two different policies:
        1. Self-consistent conditional sampling
        2. Standard optimization of any acquisition function (here: EI)
"""

class PCOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100,
                 num_self_consistency_samplings=20, num_samples=20, initial_samples=20, use_ei=False, num_ei_repeats=20,
                 num_ei_samples=1000, num_ei_variance_approx=10, interaction_dist_sample_decay=0.9, 
                 conditioning_value_quantile=1, log_hpo_runtime=False, pc_type='mspn', max_rel_ball_size=0.05, seed=0, prior_kind=None, task=None, point_prior=False) -> None:
        """
            Init PCOptimizer

        Args:
            search_space (SearchSpace): SearchSpace object
            objective (_type_): objective taking a dictionary as input and returning a BenchmarkResult object
            iterations (int, optional): Number of iterations. Defaults to 100.
            num_self_consistency_samplings (int, optional): Determines how often we sample configurations from the PC and re-evaluate them using the PC. Defaults to 20.
            num_samples (int, optional): Number of samples used in each self-consistency sample iteration. Defaults to 20.
            initial_samples (int, optional): Number of initial samples to learn PC. Defaults to 20.
            use_ei (bool, optional): Use acquisition based selection of configurations (with EI). Defaults to False.
            num_ei_repeats (int, optional): Determines how often optimization of EI is done. Best result is taken. Defaults to 20.
            num_ei_samples (int, optional): Number of samples used to optimize EI in one repeat. Defaults to 1000.
            num_ei_variance_approx (int, optional): Number of samples used to approximate variance. Defaults to 10.
            interaction_dist_sample_decay (float, optional): Decay factor of the recovery mechanism. Defaults to 0.9.
            conditioning_value_quantile (int, optional): Quantile that determines which conditioning score should be used (use only for ablations). Defaults to 1.
            log_hpo_runtime (bool, optional): Log runtime. Defaults to False.
            pc_type (str, optional): Structure to be used to learn PC. Can be 'mspn', 'parametric' or 'quantile'. Defaults to 'mspn'.
            seed (int, optional): Seed. Defaults to 0.
        """
        super().__init__(search_space, objective)
        self.curr_pc = None
        self.modified_pc = None # for interactive part and adaptation
        self.search_space = search_space
        self.objective = objective
        self._num_iterations = iterations
        self._samples_per_iter = num_samples
        self._initial_samples = initial_samples
        self._rand_state = RandomState(seed)
        self._use_eic = use_ei
        self._log_hpo_runtime = log_hpo_runtime
        self._pc_type = pc_type # can be from [mspn, parametric, quantile]
        self._max_rel_ball_size = max_rel_ball_size
        self.hpo_runtimes = []
        self.evaluations = []
        # must be recorded for logging
        self._curr_iter = 0
        self._construct_search_space_metadata()
        self._interventions = None
        self._intervention_iters: List[int] or None = None
        self._intervention_duration = None
        self._intervention = None
        self._prob_interaction_pc = 1
        self._interaction_dist_sample_decay = interaction_dist_sample_decay
        self._original_leaf_params: dict or None = None
        self._conditioning_value_quantile = conditioning_value_quantile
        self._num_ei_repeats = num_ei_repeats
        self._num_ei_samples = num_ei_samples
        self._num_ei_variance_approx = num_ei_variance_approx
        self._num_self_consistency_samplings = num_self_consistency_samplings
        self.transform = ConfigurationNumericalTransform(search_space)
        if conditioning_value_quantile != 1:
            print("WARNING: PC Optimizer is in ablation mode. For best performance set conditioning_value_quantile=1.")

        # DynaOB Adaptations
        self.use_point_prior = point_prior
        self.prior_kind = prior_kind
        self.prior_origin_path = f"{PRIOR_ORIGIN_BASE_PATH}/{task}.csv"
        self.priors = pd.read_csv(self.prior_origin_path)

    def optimize(self):
        """
            Implements optimization loop of the IBO-HPC algorithm.

        Returns:
            tuple: Tuple (cfg, score) containing the best configuration and best score (validation score).
        """
        intervene = False
        iteration = 0
        start_time = time() if self._log_hpo_runtime else None
        n_interventions_done = 0
        iterations_since_refit = 0
        refit_surrogate_next_iter = False
        while iteration < self._num_iterations:
            if iteration % 10 == 0:
                 print(f"Iteration {iteration+1}/{self._num_iterations}")

            if self.curr_pc is None:
                self._learn_init_pc()
                refit_surrogate_next_iter = True
                iteration += len(self.data)
            else:
                if iteration in self._intervention_iters:
                    refit_surrogate_next_iter = True
                    n_interventions_done +=1
                    current_cost = -1 * self.data[:, -1].max()
                    if self.prior_kind == "good":
                        relevant_configs = self.priors[self.priors["median_cost"] <=  current_cost]
                        min_cluster = relevant_configs["cluster"].min()
                        max_cluster = relevant_configs["cluster"].max()

                        if not relevant_configs.empty:
                            if min_cluster == max_cluster:
                                cluster = min_cluster
                            else:
                                cluster = self._sample_dynabo_cluster(min_cluster, max_cluster, 0.1)

                            prior_center  = self.priors[self.priors["cluster"] == cluster].sort_values("cost", ascending=False)[:1]
                            self._intervention = self._build_intervention_from_config(self.prior_kind, prior_center, n_interventions_done, iteration)
                            intervene = True
                            self._prob_interaction_pc = 1  
                    elif self.prior_kind == "medium":
                        relevant_configs = self.priors[self.priors["median_cost"] <=  current_cost]
                        min_cluster = relevant_configs["cluster"].min()
                        max_cluster = relevant_configs["cluster"].max()

                        if not relevant_configs.empty:
                            if min_cluster == max_cluster:
                                cluster = min_cluster
                            else:
                                cluster = self._sample_dynabo_cluster(min_cluster, max_cluster, 0.15)

                            relevant_configs  = self.priors[self.priors["cluster"] == cluster].sort_values("cost", ascending=False)
                            prior_center = np.random.choice(relevant_configs.index)
                            prior_center  = self.priors.loc[[prior_center]]

                            self._intervention = self._build_intervention_from_config(self.prior_kind, prior_center, n_interventions_done, iteration)
                            intervene = True
                            self._prob_interaction_pc = 1  
                    elif self.prior_kind == "misleading":
                        prior_data = deepcopy(self.priors)
                        current_incumbent = self.data[current_cost == -1 * self.data[:, -1]].flatten()[:-1]
                        hyperparameter_names = list(self.search_space.get_search_space_definition().keys())
                        incumbent_config = dict(zip(hyperparameter_names, current_incumbent))

                        closest_clusters = self.compute_closest_clusters(incumbent_config, prior_data)
                        relevant_configs = prior_data[prior_data["cluster"].isin(closest_clusters["cluster"].values)]

                        # Select cluster with lowest median cost
                        min_median_cost = relevant_configs["median_cost"].min()
                        relevant_configs = prior_data[prior_data["median_cost"] == min_median_cost]
                        prior_center = np.random.choice(relevant_configs.index)
                        prior_center  = self.priors.loc[[prior_center]]
                        self._intervention = self._build_intervention_from_config(self.prior_kind, prior_center, n_interventions_done, iteration)
                        intervene = True
                        self._prob_interaction_pc = 1  
                    elif self.prior_kind == "deceiving":
                        relevant_cluster_lower_bound = 0.95
                        relevant_cluster_upper_bound = 1.0
                        relevant_clusters = self.priors[
                            self.priors["cluster"].between(relevant_cluster_lower_bound * self.priors["cluster"].max(), relevant_cluster_upper_bound * self.priors["cluster"].max())
                        ]["cluster"].unique()
                        cluster = np.random.choice(relevant_clusters)

                        prior_center  = self.priors[self.priors["cluster"] == cluster].sort_values("cost", ascending=False)[-1:]
                        self._intervention = self._build_intervention_from_config(self.prior_kind, prior_center, n_interventions_done, iteration)
                        intervene = True
                        self._prob_interaction_pc = 1  
                    elif self.prior_kind == "no_prior":
                        self._intervention = None
                        intervene = False
                    else:
                        raise ValueError(f"Unknown prior kind: {self.prior_kind}")  

                if self._pc_type == 'parametric':
                    self._learn_pc_parametric()
                elif self._pc_type == 'quantile':
                    num_buckets = compute_bin_number(self.data.shape[0])
                    self._learn_pc_quantile(num_buckets, split_mode='linear', q_max=0.97)
                else:
                    if refit_surrogate_next_iter:
                        refit_surrogate_next_iter = False
                        iterations_since_refit = 0
                        self._learn_pc()
                    
                iterations_since_refit +=1

                configs = self._sample(intervene=intervene)
                self._evaluate(configs)

                if self._log_hpo_runtime:
                    end_time = time()
                    diff = end_time - start_time
                    self.hpo_runtimes.append(diff)
                
                #print(f"BEST: {self.data[:, -1].max()}")
                iteration += len(configs)

        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        cfg, res = self.evaluations[max_idx][:2]
        print(np.max(val_performances))
        return cfg, res
    
    
    def _sample_dynabo_cluster(self, min_cluster, max_cluster, decay):
        vals = np.arange(min_cluster, max_cluster)
        probs = np.exp(-decay * (vals - min_cluster))
        probs /= probs.sum()
        return np.random.choice(vals, p=probs[::-1])
    
    def _build_intervention_from_config(self, prior_kind: str, prior_df: pd.DataFrame, n_interventions_done: int, trial_number: int) -> Dict:
        pd1_columns = prior_df.columns.tolist()
        pd1_columns = [col for col in pd1_columns if col.startswith('config')]
        pd1_centers = prior_df[pd1_columns]

        if not self.use_point_prior:    
            search_space_bounds = {
                key: {} for key in self.search_space.get_search_space_definition().keys()
            }

            for key, val in self.search_space.get_search_space_definition().items():
                search_space_bounds[key]['min'] = val['min']
                search_space_bounds[key]['max'] = val['max']

            stds = {}
            for key, value in search_space_bounds.items():
                stds[key] = (value["max"] - value["min"]) / (5 * n_interventions_done)

            return {
                    "hps.lr_hparams.decay_steps_factor": {"dist": "gauss", "parameters": [pd1_centers["config_lr_decay_factor"], stds["hps.lr_hparams.decay_steps_factor"]]}, # Parameters is first the center, than the standard deviation 
                    "hps.lr_hparams.initial_value": {"dist": "gauss", "parameters": [pd1_centers["config_lr_initial"], stds["hps.lr_hparams.initial_value"]]},
                    'hps.lr_hparams.power' : {"dist": "gauss", "parameters": [pd1_centers["config_lr_power"], stds["hps.lr_hparams.power"]]},
                    'hps.opt_hparams.momentum' : {"dist": "gauss", "parameters": [pd1_centers["config_opt_momentum"], stds["hps.opt_hparams.momentum"]]},
            }
        else:
            return {
                "hps.lr_hparams.decay_steps_factor":  pd1_centers["config_lr_decay_factor"].iloc[0], # Parameters is first the center, than the standard deviation 
                "hps.lr_hparams.initial_value": pd1_centers["config_lr_initial"].iloc[0],
                'hps.lr_hparams.power' : pd1_centers["config_lr_power"].iloc[0],
                'hps.opt_hparams.momentum' : pd1_centers["config_opt_momentum"].iloc[0],
            }


    def compute_closest_clusters(self, incumbent_config, prior_data: pd.DataFrame):
        """Dynabo based function to compute the closest clusters"""
        # Preprocess Data
        centroid_configurations = prior_data[[col for col in prior_data.columns if col.startswith("medoid_config") or col == "cluster"]].drop_duplicates()
        centroid_configurations_only_config = centroid_configurations[[col for col in centroid_configurations.columns if col.startswith("medoid_config")]]
        centroid_configurations_only_config.columns = [col.replace("medoid_config_", "") for col in centroid_configurations_only_config.columns]

        incumbent_config = {
            "lr_decay_factor" : incumbent_config["hps.lr_hparams.decay_steps_factor"],
            "lr_initial" : incumbent_config["hps.lr_hparams.initial_value"],
            "lr_power" : incumbent_config["hps.lr_hparams.power"],
            "opt_momentum" : incumbent_config["hps.opt_hparams.momentum"],
        }

        # Create Distance Matrix
        keys = list(incumbent_config.keys())
        values = list()
        for key in keys:
            if key in incumbent_config.keys():
                values.append(incumbent_config[key])
            else:
                values.append(np.nan)
        gower_matrix_dataframe = pd.DataFrame(columns=list(keys), data=[values])
        gower_matrix_dataframe = pd.concat([gower_matrix_dataframe, centroid_configurations_only_config], ignore_index=True, axis=0)
        cat_features = [not pd.api.types.is_numeric_dtype(gower_matrix_dataframe[col]) for col in gower_matrix_dataframe.columns]
        distance_matrix = gower.gower_matrix(gower_matrix_dataframe.fillna(-1), cat_features=cat_features)

        # Select top 11 entries. The 10 closest clusters and the incumbent itself
        relevant_distances = distance_matrix[0, :]

        # Remove the incumbent itself
        closest_k_cluster_indexes = np.argsort(relevant_distances)[1:6]
        closest_k_clusters_centroids = gower_matrix_dataframe.iloc[closest_k_cluster_indexes]

        whole_entries = pd.merge(
            closest_k_clusters_centroids,
            centroid_configurations,
            left_on=list(closest_k_clusters_centroids.columns),
            right_on=[f"medoid_config_{hyperparameter}" for hyperparameter in incumbent_config.keys()],
            how="left",
        )

        return whole_entries[["cluster"]]

    def intervene(self, interventions: List[Dict], iters):
        """
            Set certain dimensions to a fixed value to obtain a conditional distribution
        """
        self._interventions = interventions
        self._intervention_iters = iters

    def adapt_to_search_space(self, new_search_space: Dict):
        """
            Adapt PC to a new search space by marginalizing unused variables out.

        """
        raise NotImplementedError()

    def _apply_point_intervention(self, data):
        # get prototype configuration to determine scope id of hyperparameter in PC.
        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    data[idx] = self._intervention[key]

        return data

    def _apply_intervention(self):

        def sample_from_condition_prior(intervention_key):
            if self._intervention[intervention_key]['dist'] == 'gauss':
                mu, std = self._intervention[intervention_key]['parameters']
                samples = np.random.normal(mu, std, self._samples_per_iter)
            elif self._intervention[intervention_key]['dist'] == 'uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = np.random.uniform(s, e, self._samples_per_iter)
            elif self._intervention[intervention_key]['dist'] == 'int_uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = np.random.randint(s, e, self._samples_per_iter)
            elif self._intervention[intervention_key]['dist'] == 'cat':
                weights = self._intervention[intervention_key]['parameters']
                probs = np.array(weights) / np.sum(weights)
                samples = np.random.choice(np.arange(len(probs)), self._samples_per_iter, p=probs)
            return samples

        # get prototype configuration to determine scope id of hyperparameter in PC.
        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        cond_samples = np.array([np.nan]*self._samples_per_iter*(len(prot_cfg) +1)).reshape(-1, len(prot_cfg) +1)
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    if isinstance(self._intervention[key], dict):
                        # dict means we have a distribution specified
                        cond_samples[:, idx] = sample_from_condition_prior(key)
                    else:
                        # else we have single point specified
                        cond_samples[:, idx] = np.repeat(self._intervention[key], self._samples_per_iter)

        return cond_samples

    def _build_dataset(self, configs, scores):
        """
            build a data matrix based on configuration and score pairs
        """
        data_matrix = []
        for i, (cfg, score) in enumerate(zip(configs, scores)):
            features = self.transform.transform_configuration(cfg)
            features.append(score.val_performance)
            data_matrix.append(features)
            self.evaluations.append((cfg, score, self._curr_iter))
        return np.array(data_matrix)
    
    def _learn_init_pc(self):
        """
            First, generate a set of random samples from the search space and evaluate them.
            Then, build a dataset and learn a PC over the joint space of hyperparameters and evaluation score.
        """
        configs = self.search_space.sample(size=self._initial_samples)
        scores = [self.objective(cfg) for cfg in configs]
        self.data = self._build_dataset(configs, scores)
        if self._pc_type == 'parametric':
            self._learn_pc_parametric()
        elif self._pc_type == 'quantile':
            num_buckets = compute_bin_number(self.data.shape[0])
            self._learn_pc_quantile(num_buckets, split_mode='linear', q_max=0.95)
        else:
            self._learn_pc()

    def _learn_pc(self):
        """
            Learn a MSPN over the joint space of hyperparameters and evaluations.
        """
        self._construct_search_space_metadata()
        self.ctxt.add_domains(self.data)
        mu_hp = max(5, self.data.shape[0] // 100)
        # NOTE: If this fails with memory error, try rounding the performances
        try:
            self.curr_pc = learn_mspn(self.data, self.ctxt, min_instances_slice=10, threshold=0.2, n_clusters=2)
        except ValueError:
            # sometimes learning fails due convergence failure of RDC. Try again with less restrictions
            self.curr_pc = learn_mspn(self.data, self.ctxt, min_instances_slice=40, threshold=0.5, n_clusters=2)

    def _learn_pc_parametric(self):
        """
            Learn a parametric SPN (Gaussian leaves).
            This requires all hyperparameters to be continuous.
        """
        params = [Gaussian] * (self.data.shape[1])
        meta_types = [MetaType.REAL] * (self.data.shape[1])
        ctxt = Context(meta_types=meta_types, parametric_types=params)
        X_joint = self.data
        ctxt.add_domains(X_joint)
        try:
            self.curr_pc = learn_parametric(X_joint, ctxt, min_instances_slice=5, threshold=0.3)
        except ValueError:
            self.curr_pc = learn_parametric(X_joint, ctxt, min_instances_slice=15, threshold=0.5)

    def _learn_pc_quantile(self, buckets=20, split_mode='linear', min_samples_per_bucket=10, q_max=0.99):
        """
            Learn a PC using the quantile approach: First, we bucketize the evaluation scores.
            Depending on 'split_mode', this happens by splitting the sorted array of evaluation scores
            in equally sized slices, by performing recursive splitting of the evaluation space or both.

            # Equal Splits
            If we split equally, the evaluation scores get sorted and the is split s.t. each bucket contains
            the same number of samples. Since the configurations remain assigned to their score, we split
            the data space into bad, medium and good samples. For each split (possibly more than 3 as illustrated here),
            we fit a PC and form a mixture over all PCs in the end.

            # Recursive Splits
            We start by computing the 50%-quantile, split the space into "good" and "bad", pick the "good" part of the 
            data and repeat this procedure until the smallest bucket contains 'min_samples_per_bucket' samples. 
            Then, for each bucket a PC is learned and a mixture of PCs is formed as a final model.

            # Equal Splits + Recursive Splits
            We first apply recursive splittint of the data space and then perform equal splitting with all 
            "bad" parts of the search space. Again, a mixture of PCs is formed in the end to obtain the full model.

        Args:
            buckets (int, optional): Number of buckets to split the data into. Defaults to 20.
            split_mode (str, optional): Mode of splitting. Can be 'linear', 'binary_tree' or 'bt+lin'. Defaults to 'linear'.
            min_samples_per_bucket (int, optional): Minimum samples required per bucket. Defaults to 10.
            q_max (float, optional): Maximum quantile used if split_mode is 'binary_trees' or 'bt+lin'. Defaults to 0.99.
        """
        self._construct_search_space_metadata()
        self.ctxt.add_domains(self.data)
        y = self.data[:, -1]
        idx = np.argsort(y).flatten()
        if split_mode == 'linear':
            indices = np.array_split(idx, buckets)
        elif split_mode == 'binary_tree':
            _, indices = create_buckets(y[idx], idx, min_samples=min_samples_per_bucket)
        elif split_mode == 'bt+lin':
            _, indices = create_buckets(y[idx], idx, min_samples=min_samples_per_bucket)
            first_idx = indices[0]
            first_indices = np.array_split(first_idx, buckets)
            indices = first_indices + indices[1:]
        elif split_mode == 'quantile':
            _, indices = create_quantile_buckets(y, q_max=q_max)
        spns = []
        weights = []
        for idx_chunk in indices:
            subset = self.data[idx_chunk]
            weights.append(len(idx_chunk))
            try:
                spn = learn_mspn(subset, self.ctxt, min_instances_slice=5, threshold=0.2, n_clusters=2)
            except ValueError:
                # sometimes learning fails due convergence failure of RDC. Try again with less restrictions
                spn = learn_mspn(subset, self.ctxt, min_instances_slice=40, threshold=0.5, n_clusters=2)
            spns.append(spn)

        sum = Sum()
        scope = spns[0].scope
        sum.scope = scope
        weights = np.array(weights) / np.sum(weights)
        sum.weights = weights
        sum.children = spns
        spn = assign_ids(sum)
        self.curr_pc = spn
    
    def _get_conditioning_value(self):
        """
            Only used for ablation studies to analyze importance of conditioning on the best value.
            If optimizer is running in normal mode, set self._conditioning_value_quantile=1.
        """
        if self._conditioning_value_quantile == 1:
            return self.data[:, -1].max()
        else:
            return np.quantile(self.data[:, -1], self._conditioning_value_quantile)

    def _sample(self, intervene=False):
        """
            Sample from the PC to obtain the next configuration(s) to test.
            Depending on whether `use_ei` is True or False, the next configurations
            are selected using the EI acquisition function or via self-consistent
            conditional sampling.

            # EI
            If EI is used, 'num_ei_repeats'*'num_ei_samples' random samples are drawn from the search space
            and the best 'num_ei_repeats' ones are kept and tested.

            # Self-consistent Conditional Sampling
            If the self-consistent conditional sampling is used, 'num_self_consistency_samplings' rounds of 
            conditional sampling are performed as follows: In each round, 'num_samples' configurations are 
            drawn from the PC by conditioning the PC on the best evaluation score obtained so far. Then, each
            configuration is fed back to the PC which in turn predicts the expected evaluation score each
            sampled configuration will achieve. In each sampling round, we keep the configuration with the
            highest expected evaluation score.

        Args:
            intervene (bool, optional): Determines if user interaction is used during sampling. Defaults to False.

        Returns:
            _type_: Set of configurations tested next.
        """        
        cond_array = np.array([np.nan] * self._samples_per_iter * self.data.shape[1]).reshape(-1, self.data.shape[1])
        #print(self._get_conditioning_value())
        cond_array[:, -1] = self._get_conditioning_value()
        if intervene:
            accept_intervention = np.random.choice([1, 0], p=[self._prob_interaction_pc, 1-self._prob_interaction_pc])
            self._prob_interaction_pc *= self._interaction_dist_sample_decay
            if accept_intervention == 1:
                cond_array = self._apply_intervention()
            cond_array[:, -1] = self._get_conditioning_value()
        
            # Randomly mask 1 to all intervention dimensions to allow PC to have influence
            #always set prior on learing rate and momentum
            use_the_best_hyperparameters = False
            if use_the_best_hyperparameters:
                # set dim 0 = nan and dim 2 = nan
                cond_array[:, 0] = np.nan
                cond_array[:, 2] = np.nan
            else:
                for i in range(len(cond_array)):
                    n_maskedims = np.random.randint(1, len(self._intervention.keys()) +1, len(cond_array))
                    mask_dims = np.random.choice(range(len(self._intervention.keys())), n_maskedims[i], replace=False)
                    for position in mask_dims:
                        cond_array[i, position] = np.nan

        # EI sampling¨        
        if self._use_eic:
            sampled_configs = self._ei()
        # Conditional sampling with self-reensurance
        else:
            best_configs = []
            for i in range(self._num_self_consistency_samplings):
                samples_ = sample_instances(self.curr_pc, np.array(cond_array), np.random.RandomState(i))
                re_ensurance_cond = np.copy(samples_)
                re_ensurance_cond[:, -1] = np.nan
                y_pred = sample_instances(self.curr_pc, re_ensurance_cond, np.random.RandomState(i))[:, -1].flatten()
                samples_[:, -1] = y_pred
                idx = np.argmax(y_pred).flatten()[0]
                best_configs.append(samples_[idx])
            samples = np.array(best_configs)
            y_ = samples[:, -1]
            idx = [np.argmax(y_)]
            samples = samples[idx]
            sampled_configs = samples[:, :-1]

        #sampled_configs = sample_from_ball(sampled_configs, self.search_space, self.ball_radius, sampled_configs.shape[0])
        # We removed this after talking to the papers of the original papers. They do not use this in their final version.
        
        for i, (_, v) in enumerate(self.search_space.get_search_space_definition().items()):
            if v['dtype'] == 'float':
                for j in range(sampled_configs.shape[0]):
                    sampled_configs[j, i] = max(sampled_configs[j, i], v['min'])
                    sampled_configs[j, i] = min(sampled_configs[j, i], v['max'])
        return sampled_configs

    def _evaluate(self, sampled_configs):
        """
            For all sampled configurations, invoke the evaluation procedure.
            The results are added to the dataset used to fit the PC.

        Args:
            sampled_configs (_type_): Set of sampled configurations
        """
        evaled_samples = []
        for i in range(len(sampled_configs)):
            sample = list(sampled_configs[i])
            # if already sampled, skip
            #if np.any(np.all(self.data[:, :-1] == np.array(sample), axis=1)):
            #    continue
            config_dict = self.transform.inv_transform_configuration(sample)
            evaluation = self.objective(config_dict)
            if evaluation is not None:
                self.evaluations.append((config_dict, evaluation, self._curr_iter))
                sample.append(evaluation.val_performance)
                evaled_samples.append(sample)
        evaled_samples = np.array(evaled_samples)
        if evaled_samples.shape[0] > 0:
            # may be that we have seen all sampled configs already
            self.data = np.concatenate((self.data, evaled_samples))
    
    def _ei(self):
        best_samples = []
        for j in range(self._num_ei_repeats):
            eic_samples = self.search_space.sample(size=self._num_ei_samples)
            eic_samples = np.array([self.transform.transform_configuration(cfg) for cfg in eic_samples])
            nan_column = np.full((eic_samples.shape[0], 1), np.nan)
            cond_matrix = np.hstack((eic_samples, nan_column))
            pred_mat = []
            for i in range(self._num_ei_variance_approx):
                preds = sample_instances(self.curr_pc, cond_matrix, np.random.RandomState(i))[:, -1]
                pred_mat.append(preds)
            pred_mat = np.array(pred_mat)
            mean_preds = np.mean(pred_mat, axis=0)
            mean_std = np.std(pred_mat, axis=0)
            improve = self._get_conditioning_value() - 0.01 - mean_preds
            scaled = improve / mean_std
            cdf = scipy.stats.norm.cdf(scaled)
            pdf = scipy.stats.norm.pdf(scaled)
            exploit = improve * cdf
            explore = mean_std * pdf
            best = np.argsort(explore + exploit)
            best_samples.append(eic_samples[best[0]])
        return np.array(best_samples)
    
    def _construct_search_space_metadata(self):
        search_space_definition = self.search_space.get_search_space_definition()
        meta_types = [val['type'] for val in search_space_definition.values()]
        meta_types += [MetaType.REAL]
        self.hyperparam_names = list(search_space_definition.keys())
        self.ctxt = Context(meta_types=meta_types)

        # set radius for sampling
        self.ball_radius = []
        for idx, (name, hp_def) in enumerate(search_space_definition.items()):
            if hp_def['dtype'] == 'float':
                if isinstance(self._max_rel_ball_size, list):
                    rel_rad = self._max_rel_ball_size[idx]
                else:
                    rel_rad = self._max_rel_ball_size
                
                if hp_def['is_log']:
                    r = (np.log(hp_def['max']) - np.log(hp_def['min'])) * rel_rad
                else:
                    r = (hp_def['max'] - hp_def['min']) * rel_rad
                self.ball_radius.append(r)
            else:
                # if discrete HP, interprete the relative max ball size as a probability to sample randomly from the dimension
                if isinstance(self._max_rel_ball_size, list):
                    self.ball_radius.append(self._max_rel_ball_size[idx])
                else:
                    self.ball_radius.append(self._max_rel_ball_size)

    def _print_acc(self):
        evidence = np.copy(self.data)
        evidence[:, -1] = np.nan
        y_test = self.data[:, -1].flatten()

        preds = []
        for i in range(20):
            pc_preds = sample_instances(self.curr_pc, evidence, np.random.RandomState(i))
            preds.append(pc_preds[:, -1].flatten())
        pc_preds = np.array(preds).mean(axis=0)
        #pc_preds = mpe(spn, evidence)[:, -1].flatten()
        pc_r2 = r2_score(y_test, pc_preds.flatten())
        print(pc_r2)

    @property
    def history(self):
        if self._log_hpo_runtime:
            evaluations = []
            optim_cost = np.sum(self.hpo_runtimes)
            for cfg, eval_res, it in self.evaluations:
                eval_res.set_optim_cost(optim_cost)
                evaluations.append((cfg, eval_res, it))
            return evaluations
        return self.evaluations
    
def sample_from_ball(center, search_space: SearchSpace, radius, n_samples):
    """
    Sample random points uniformly from a ball around a given center.
    
    Parameters:
        center (array-like): Coordinates of the center of the ball.
        search_space (SearchSpace): Search space of the current HPO task.
        radius (list or float): Radius of the ball per dimension.
        n_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Array of shape (n_samples, dimension) with sampled points.
    """
    if isinstance(radius, list):
        radius = np.array(radius)
    points = []

    search_space_def = search_space.get_search_space_definition()

    # only apply ball sampling to real dimensions
    real_dims = [i for i, v in enumerate(search_space_def.values()) if v['dtype'] == 'float']
    other_dims = [(name, i) for i, (name, v) in enumerate(search_space_def.items()) if v['dtype'] != 'float']

    for i in range(n_samples):
        if len(real_dims) > 0:
            # Generate a random direction (unit vector) in `dimension` space
            direction = np.random.normal(size=len(real_dims))
            direction /= np.linalg.norm(direction)
            
            # Generate a random distance from the center, scaled to fit inside the radius
            distance = np.random.uniform(0, 1) ** (1 / len(real_dims)) * radius[real_dims]
            
            # Compute the point
            center[i, real_dims] = center[i, real_dims] + direction * distance
        for name, j in other_dims:
            coin = np.random.choice([0, 1], p=[1-radius[j], radius[j]])
            if coin == 1:
                choices = np.arange(0, len(search_space_def[name]['allowed']))
                rnd_cfg = np.random.choice(choices)
                center[i, j] = rnd_cfg
        points.append(center[i])
    return np.array(points)