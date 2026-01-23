import argparse
import os
from datetime import datetime
from multiprocessing import Process

import numpy as np
import sklearn
from rtpt import RTPT

from ihpo.experiments import (
    PD1Experiment,
    PD1InteractiveExperiment,
    PD1TransferExperiment,
)
from ihpo.utils import SEEDS, create_dir

def get_experiment(args, seed, prior_kind, point_prior):
    if args.exp == 'pd1':
        experiment = PD1Experiment(args.optimizer, args.task, seed=seed, prior_kind=prior_kind)
    elif args.exp == 'pd1_int':
        experiment = PD1InteractiveExperiment(args.optimizer, prior_kind=prior_kind, task=args.task, interaction_idx=args.interaction_idx, seed=seed, point_prior=point_prior)
    else:
        raise ValueError(f'No such experiment: {args.exp}. Must be hpo, nas101, nas201, transnas or jahs.')
    return experiment


def run_experiment(args, seed_idx, prior_kind, point_prior):
    exp_counter = 0
    rt = RTPT('JS', 'IHPO', 1)
    rt.start()
    # set random seed for each experiment
    #seed = np.random.choice(np.array(SEEDS))
    seed = SEEDS[seed_idx]
    np.random.seed(seed)
    sklearn.random.seed(seed)
    
    exp_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    file_name = f'{args.exp}_{args.task}_{exp_time}_{seed}.csv'
    log_file = os.path.join(args.log_dir, file_name)
    experiment = get_experiment(args, seed, prior_kind, point_prior)
    try:
        # PC fails sometimes, ignore these runs
        experiment.run()
        experiment.save(log_file)
        exp_counter += 1
        rt.step()
    except Exception as e:
        raise e
        print(f"Experiment failed with: {e}")

point_prior = True
for task in ["lm1b_transformer_2048",]:
    for prior_kind in ["good", "medium", "misleading", "deceiving",]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-experiments', default=30, type=int)
        parser.add_argument('--optimizer', default='pc', type=str)
        parser.add_argument('--exp', default='pd1_int', type=str)
        parser.add_argument('--task', default=task)
        parser.add_argument('--tasks', default=[task], nargs='+')
        parser.add_argument('--num-procs', default=5, type=int)
        parser.add_argument('--seed-offset', type=int, default=0)
        parser.add_argument('--num-prior-runs', type=int, default=5, help='Number of runs considered per task in HTL setting.')
        parser.add_argument('--handle-invalid-configs', action='store_true', help='Provides fall back value for invalid configurations. Only applies to some experiments.')
        parser.add_argument('--interaction-idx', nargs='+', type=int, default=[13, 23, 33, 43])
        if prior_kind != "no_prior":
            if not point_prior:
                parser.add_argument('--log-dir', default=f'./pd1_experiments/with_prior/{task}/{prior_kind}', type=str)
            elif point_prior:
                parser.add_argument('--log-dir', default=f'./pd1_experiments/with_point_prior/{task}/{prior_kind}', type=str)
        else:
            parser.add_argument('--log-dir', default=f'./pd1_experiments/no_prior/{task}/', type=str)
    
        args = parser.parse_args()

        #setup_environment()
        create_dir(args.log_dir)
        if args.optimizer.startswith('pc'):
            print("PC optimizer not parallelizable. Run in single thread mode.")
            exp_counter = 0
            while exp_counter < args.num_experiments:
                run_experiment(args, args.seed_offset + exp_counter, prior_kind, point_prior)
                exp_counter += 1
        else:
            num_procs = min(args.num_procs, args.num_experiments) # min ensures thtat there is at least one batch of processes
            num_batches = int(round(args.num_experiments / num_procs))

            for b in range(num_batches):
                print(f"Start batch {b+1}/{num_batches} of experiments")
                seed_offset = b*num_procs
                processes = [Process(target=run_experiment, args=(args, seed_offset+i)) for i in range(num_procs)]
                for p in processes:
                    p.start()

                for p in processes:
                    p.join()