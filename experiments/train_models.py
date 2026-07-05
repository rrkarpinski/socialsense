#!/usr/bin/env python3
"""
Simple training script - just extracts the config dict and runs training.
"""
import os
import sys

# Disable tqdm progress bars - use print statements instead
os.environ['TQDM_DISABLE'] = '1'

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import datetime
import time
import json
from transformers import get_linear_schedule_with_warmup
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('..')

from models.buffers import NaiveRehearsalBuffer
from data_processing.data_utils import (
    get_transform, serialize_transform, get_domain_dataloaders, 
    pool_domain_dataloaders, get_crossvalidation_domain_loaders, 
    IMAGENET_NORM, get_domain_dataloaders_from_hdf5
)
from models.training_utils import heuristic_dualbranch_batch, unified_train_loop
from models.heuristicSplitModel import DualBranchModel


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    # Import the config from config file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        # Import the config module
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        device = config_module.device
        domains = config_module.domains
        testing_scenarios = config_module.testing_scenarios
        
        # Get other training parameters if they exist
        num_workers = getattr(config_module, 'num_workers', 0)
        pin_memory = getattr(config_module, 'pin_memory', True)
        persistent_workers = getattr(config_module, 'persistent_workers', False)
        persistent_workers = persistent_workers if num_workers > 0 else False
        batch = getattr(config_module, 'batch', (32, 64, 64))
        branch = getattr(config_module, 'branch', 'mobilenetv2')
        seed = getattr(config_module, 'seed', 42)
        grad_clipping = getattr(config_module, 'grad_clipping', True)
        branch_norm = getattr(config_module, 'branch_norm', True)
        earlystopping = getattr(config_module, 'earlystopping', True)
        epochs = getattr(config_module, 'epochs', 20)
        alldomains = getattr(config_module, 'alldomains', False)
        dropout_rate = getattr(config_module, 'dropout_rate', 0.3)
        lr = getattr(config_module, 'lr', 1e-3)
        
        print(f"Loaded config from: {config_file}")
        print(f"Num workers: {num_workers}")
        print(f"Pin memory: {pin_memory}")
        print(f"Found {len(testing_scenarios)} scenarios to train\n")
    else:
        print("ERROR: Please provide a config file as argument")
        print("Usage: python train_models.py <config_file.py>")
        sys.exit(1)
    
    set_seed(seed)
    
    # Main training loop
    for name, (ablation, buffer_size, train_val_path, test_path, hdf5_path, transform_list, img_path_cols, scene_as_label) in testing_scenarios.items():
    # Instead of looping through all scenarios:
# for name, (...) in testing_scenarios.items():
# Use:
    # name = config_module.selected_scenario
    # (ablation, buffer_size, train_val_path, test_path, hdf5_path, transform_list, img_path_cols, scene_as_label) = testing_scenarios[name]
# ... rest of your script unchanged ...

# (keep the existing DataLoader logic!)

    
        train_df = pd.read_pickle(train_val_path) if train_val_path is not None else None
        test_df = pd.read_pickle(test_path) if test_path is not None else None
        hdf5_dataset_path = hdf5_path
        
        set_seed(seed)
        
        # Get dataloaders
        if hdf5_path:
            domain_dataloaders = get_domain_dataloaders_from_hdf5(
                hdf5_path=hdf5_dataset_path, 
                domains=domains, 
                img_path_cols=img_path_cols, 
                batch_sizes=batch,
                num_workers=num_workers, 
                set_first_element_as_domain_label=scene_as_label,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory
            )
        else:
            domain_dataloaders = get_domain_dataloaders(
                train_df, 
                seed=seed, 
                batch_sizes=batch, 
                img_path_cols=img_path_cols, 
                transforms=transform_list, 
                num_workers=num_workers, 
                include_test=test_df, 
                set_first_element_as_domain_label=scene_as_label,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )
        
        if alldomains:
            domain_dataloaders = pool_domain_dataloaders(domain_dataloaders)
            domains_to_use = ['AllDomains']
        else:
            domains_to_use = domains
        
        print(f"\nTesting: {name} Ablation: {ablation}")
        
        # Setup model configuration
        setup = {'branch': branch}
        if ablation == 'nomask':
            setup['ablation'] = 'focus'
        elif ablation == 'onlysoc':
            setup['ablation'] = 'scene'
        elif ablation == 'onlyenv':
            setup['ablation'] = 'focus'
        else:
            setup['ablation'] = False
        
        if scene_as_label:
            setup['scene'] = 'label'
        
        # Initialize model
        model = DualBranchModel(dropout_rate=dropout_rate, setup=setup, branch_norm=branch_norm)
        dual_model = model.to(device)
        
        # Setup optimizer
        trainable_params = [p for p in dual_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        # Setup buffer
        buffer = NaiveRehearsalBuffer(buffer_size=buffer_size)
        
        # Create experiment name
        exp_name = f"{name}_buffer={buffer_size}_epochs={epochs}_seed={seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Scheduler
        scheduler = (get_linear_schedule_with_warmup, 0.05)
        refresh_optimiser = False
        
        # Loss function
        dualbranch_kwargs = {
            'mse_criterion': nn.MSELoss(),
        }
        
        # Save configuration
        config = {
            "model": {
                "name": str(type(model)),
                "setup": setup,
                "branch_norm":branch_norm,
                "dropout_rate":dropout_rate,
            },
            "buffer": str(type(buffer)),
            "buffer_size": buffer_size,
            "optimizer": str(type(optimizer)),
            "learning_rate": lr,
            "scheduler": {
                "type": "transformers.get_linear_schedule_with_warmup",
                "warmup": scheduler[1],
            },
            "device": str(device),
            "loss": str(dualbranch_kwargs['mse_criterion']),
            "training_mode": "noCL" if alldomains else "CL",
            "seed": seed,
            "domains_order": domains,
            "train_val_df": train_val_path,
            "test_df": test_path,
            "hdf5_dataset_path": hdf5_dataset_path,
            "input_columns": img_path_cols,
            "input_transforms": [serialize_transform(x) for x in transform_list],
            "dataloader": {
                "batch_size": batch,
                "num_workers":num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
            },
            "epochs": epochs,
            "early_stopping": {
                "patience":15,
                "delta":1e-3,
            },
            "gradient_clipping": grad_clipping,
            "refresh_optimiser": refresh_optimiser,
        }
        
        with open(f"../checkpoints/{exp_name}_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        unified_train_loop(
            model=dual_model,
            domains=domains_to_use,
            domain_dataloaders=domain_dataloaders,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            batch_fn=heuristic_dualbranch_batch,
            batch_kwargs=dualbranch_kwargs,
            num_epochs=epochs,
            exp_name=exp_name,
            gradient_clipping=grad_clipping,
            checkpoint_dir="../checkpoints",
            validation_set='val',
            scheduler=scheduler,
            refresh_optimiser=refresh_optimiser,
            early_stopping=earlystopping,
        )

        print("=" * 80)
        print("All training completed!")
        print("=" * 80)
