import torch
from tqdm.auto import tqdm
# from tqdm import tqdm,trange
import numpy as np
import pickle
import datetime
import os
import torch
from typing import Dict, Any, Tuple
TQDM_DISABLED = bool(os.environ.get("TQDM_DISABLE", "0") == "1")

def heuristic_dualbranch_batch(model, batch, device, **kwargs):
    inputs, labels, domain_labels = batch
    inputs = tuple(x.to(device) for x in inputs)
    labels, domain_labels = labels.to(device), domain_labels.to(device)
    mse_criterion = kwargs['mse_criterion']

    outputs = model(*inputs)

    loss = mse_criterion(outputs['output'], labels)
    
    metrics = {}
    return loss, metrics

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch[-1].shape[0]
            loss, _ = heuristic_dualbranch_batch(model, batch, device, mse_criterion=criterion)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        val_loss = total_loss/total_samples
    return val_loss

def cross_domain_validation(model, domain_dataloaders, criterion, device, validation_set='val'):
    results = {}
    for domain, loaders in domain_dataloaders.items():
        val_loader = loaders[validation_set]
        val_loss = evaluate_model(model, val_loader, criterion, device)
        results[domain] = val_loss
    return results

def average_metrics(metrics_list):
    # metrics_list: list of dicts, each dict contains metrics for a batch
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg_metrics = {}
    for k in keys:
        avg_metrics[k] = float(np.mean([m[k] for m in metrics_list if k in m]))
    return avg_metrics

def collect_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and not name.startswith("backbone"):
            module = name.split('.')[0]
            norm = param.grad.norm(2).item()
            if module not in grad_norms:
                grad_norms[module] = []
            grad_norms[module].append(norm)
    # Take mean per module
    grad_norms = {k: float(np.mean(v)) for k, v in grad_norms.items()}
    return grad_norms

class EarlyStopping:
    def __init__(self, checkpoint_dir: str, model_name: str, patience: int, verbose: bool = True, delta: float = 1e-3):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.patience_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any], epoch=int) -> bool:
        score = -val_loss  # Since lower val_loss is better, invert for comparison

        if (self.best_score is None) or (score >= self.best_score + self.delta):
            self.best_score = score  
            if self.verbose:
                self.val_loss_min = float(val_loss)
                print(f"Validation loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving checkpoint.")
            self._save_checkpoint(model, optimizer, history, epoch)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f"No improvement. EarlyStopping patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

    def _save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any], epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
        torch.save(checkpoint, path)

    def restore_best_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: Dict[str, Any]) -> int:
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history.clear()
        history.update(checkpoint["history"])
        return checkpoint['epoch']


from contextlib import contextmanager
import time

@contextmanager
def timing(history, key):
    """
    To use, add 
    'timings' to history dictionary
    and
    with timing(history['timings'], 'description'):
    around anything to time
    """
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    history.setdefault(key, []).append(duration)

def unified_train_loop(
    model, domains, domain_dataloaders, buffer, optimizer, device,
    batch_fn, batch_kwargs, num_epochs, exp_name, 
    gradient_clipping=False, collect_tsne_data=False, restart={}, 
    eval_buffer=False, checkpoint_dir="../checkpoints", validation_set='val', scheduler=None, refresh_optimiser=False, early_stopping=True,
):
    scaler = torch.amp.GradScaler('cuda') if torch.device(device).type == "cuda" else None
    if scheduler is not None:
        scheduler, warmup = scheduler
    
    start_domain_idx = 0
    history = {
        'train_epoch_loss': [],
        'val_epoch_loss': [],
        'train_epoch_metrics': [],
        # 'cross_domain_val': [],
        'grad_norms': [],
        # 'timings': {},
    }
    if eval_buffer:
        history['val_buffer_epoch_loss']=[]
    
    if restart:
        # Populate history
        history = restart.get('history', {})
        # Populate buffer
        start_domain_idx = domains.index(restart['domain'])
        for domain_idx, current_domain in enumerate(domains[:start_domain_idx]):
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset) 
        print(f"Restarting from domain {restart['domain']} index {start_domain_idx}")
        print(f"Buffer: {buffer.get_domain_distribution()}")         
        
    domains_iter = tqdm(domains[start_domain_idx:], desc="Total training", position=0, disable=TQDM_DISABLED)
    for domain_idx, current_domain in enumerate(domains_iter, start=start_domain_idx):
        if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Starting domain {current_domain}")
        if bool(buffer):
            train_loader = buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain]['train'])
        else:
            train_loader = domain_dataloaders[current_domain]['train']
        if eval_buffer:
            eval_loader = eval_buffer.get_loader_with_replay(current_domain, domain_dataloaders[current_domain][validation_set])
            
        len_dataloader = len(train_loader)

        # Initialize new optimiser like the previous one for each domain
        if refresh_optimiser:
            optimizer = type(optimizer)(optimizer.param_groups)

        if scheduler is not None:
            total_training_steps = num_epochs * len_dataloader
            warmup_steps = int(warmup * total_training_steps)
            lr_scheduler = scheduler(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
        if early_stopping:
            model_name = exp_name + f"_domain{current_domain}"
            early_stopper = EarlyStopping(checkpoint_dir=checkpoint_dir, model_name=model_name, patience=15, verbose=False, delta=1e-3)
        
        epoch_iter = tqdm(range(num_epochs), desc=f"Domain {current_domain}", position=1, disable=TQDM_DISABLED)
        for epoch in epoch_iter:
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Starting epoch {epoch}/{num_epochs}")
            model.train()
            epoch_loss = 0.0
            samples = 0
            batch_metrics_list = []
            
            # batch_iter = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", position=2, leave=False)
            for batch_idx, batch in enumerate(train_loader):
            # for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Current epoch {epoch}", position=2, leave=False, disable=TQDM_DISABLED)):
                if not batch_kwargs.get('alpha'):
                    p = (epoch * len_dataloader + batch_idx) / (num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    alpha = batch_kwargs['alpha']

                optimizer.zero_grad()

                if torch.device(device).type == "cuda":
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss, metrics = batch_fn(model, batch, device, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                    scaler.scale(loss).backward()
                    if gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    old_scaler = scaler.get_scale()
                    scaler.update()
                    new_scaler = scaler.get_scale()
                    if new_scaler >= old_scaler and scheduler is not None:
                        lr_scheduler.step()
                else:
                    loss, metrics = batch_fn(model, batch, device, **{**batch_kwargs, 'current_domain': current_domain, 'alpha':alpha})
                    loss.backward()
                    if gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        lr_scheduler.step()    
                    
                metrics.setdefault('lrs', []).append(optimizer.param_groups[0]['lr'])

                batch_size = batch[-1].size(0)
                epoch_loss += loss.item() * batch_size
                samples += batch_size
                batch_metrics_list.append(metrics)
            
            # batch_iter.close()
            # tqdm._instances.clear()

            avg_epoch_loss = epoch_loss / samples
            history['train_epoch_loss'].append(avg_epoch_loss)
            # Average batch metrics for this epoch
            avg_metrics = average_metrics(batch_metrics_list)
            history['train_epoch_metrics'].append(avg_metrics)
            
            # Collect gradients
            grad_norms = collect_gradients(model)
            history['grad_norms'].append(grad_norms)
            
            # Validation on current domain
            val_loss = evaluate_model(model, domain_dataloaders[current_domain][validation_set], batch_kwargs['mse_criterion'], device)    
            history['val_epoch_loss'].append(val_loss)
            if eval_buffer:
                val_loss_buffer = evaluate_model(model, eval_loader, batch_kwargs['mse_criterion'], device)
                history['val_buffer_epoch_loss'].append(val_loss_buffer)
            
            with open(f"{checkpoint_dir}/{exp_name}_history.pkl", "wb") as f:
                pickle.dump(history, f)
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: History pickle updated")

            if early_stopping:
                stop = early_stopper(val_loss, model, optimizer, history, epoch)

                if stop or (epoch == num_epochs-1): 
                    best_epoch = early_stopper.restore_best_checkpoint(model, optimizer, history)
                    print(f"Early stopping triggered at domain {current_domain} epoch {epoch}. Model restored to epoch {best_epoch}")
                    break

        epoch_iter.close()

        # Instead of batchwise average do cross domain validation on inference on all test samples
        # Cross-domain validation (after each domain)
        # cross_val = cross_domain_validation(model, domain_dataloaders, batch_kwargs['mse_criterion'], device=device, validation_set=validation_set)
        # history['cross_domain_val'].append(cross_val)
        
        # Handle saving through EarlyStopper
        # Only save last model per domain to save space
        if not early_stopping:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f"{checkpoint_dir}/{exp_name}_domain{current_domain}_epoch{epoch}.pt")
            if TQDM_DISABLED: print(f"[{exp_name}]\t{datetime.datetime.now()}: Checkpoint saved at epoch {epoch}")
          
        if bool(buffer):
            buffer.update_buffer(current_domain, domain_dataloaders[current_domain]['train'].dataset)
        if eval_buffer:
            eval_buffer.update_buffer(current_domain, domain_dataloaders[current_domain][validation_set].dataset)
    
    domains_iter.close()

    # Save all domain models in one gzip
    # archive_domain_models(exp_name, checkpoint_dir)
    return history


import os
import tarfile
import gzip
from pathlib import Path

def archive_domain_models(core_name, checkpoints_dir='../checkpoints'):
    checkpoints_dir = Path(checkpoints_dir)

    domain_files = list(checkpoints_dir.glob(f"{core_name}_domain*.pt"))
    if len(domain_files) != 6:
        raise FileNotFoundError(f"{len(domain_files)} domain .pt files found for {core_name}")

    config_files = list(checkpoints_dir.glob(f"{core_name}_config.json"))
    if len(config_files) != 1:
        raise FileNotFoundError(f"{len(config_files)} config files found for {core_name}_config.json")

    all_files = domain_files + config_files

    num_steps = len(all_files) + 1  # each file added + gzip compress step
    with tqdm(total=num_steps, desc="Archiving and compressing") as pbar:
        tar_name = checkpoints_dir / f"{core_name}.tar"
        with tarfile.open(tar_name, "w") as tar:
            for file in all_files:
                tar.add(file, arcname=file.name)
                pbar.update(1)

        gzip_name = checkpoints_dir / f"{core_name}.tar.gz"
        with open(tar_name, 'rb') as f_in, gzip.open(gzip_name, 'wb') as f_out:
            f_out.writelines(f_in)
        pbar.update(1)

    os.remove(tar_name)
