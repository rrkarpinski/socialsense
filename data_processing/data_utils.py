import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import ast
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py
from tqdm import tqdm
import re

LABEL_COLS = [
    "Vaccum Cleaning", "Mopping the Floor", "Carry Warm Food",
    "Carry Cold Food", "Carry Drinks", "Carry Small Objects",
    "Carry Large Objects", "Cleaning", "Starting a conversation"
]
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
NORM_VALUES=(mean,std)

IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# MANNERSDB+ and OFFICE-MANNERSDB normalization mean=[0.640, 0.516, 0.387] std=[0.237, 0.244, 0.240]
CUSTOM_NORM = ([0.640, 0.516, 0.387], [0.237, 0.244, 0.240])
DEPTH_NORM = ([2.142], [0.931])

def get_transform(resize, normalize_mean_std=None):
    """
    Return a customised default torchvision transform like
    transform = T.Compose([
        T.Resize((144,256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    """
    transforms = [
        T.Resize(resize),
        T.ToTensor(),
    ]
    if normalize_mean_std is not None:
        transforms.append(
             T.Normalize(mean=normalize_mean_std[0], std=normalize_mean_std[1])
        )
    return T.Compose(transforms)

def serialize_transform(transform):
    steps = []
    for sub in transform.transforms:
        cls_name = sub.__class__.__name__
        if isinstance(sub, T.Resize):
            args = {"size": sub.size}
        elif isinstance(sub, T.Normalize):
            args = {"mean": sub.mean, "std": sub.std}
        elif isinstance(sub, T.ToTensor):
            args = {}
        else:
            raise NotImplementedError(f"Serialization not implemented for {cls_name}")
        steps.append({"type": cls_name, "args": args})
    return steps

def _parse_resize_param(transform_str):
    # Look for size=(number, number)
    match = re.search(r'size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', transform_str)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return (width, height)
    else:
        print("Size not found")
        return None

def _parse_normalize_params(transform_str):
    # Match something like: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_match = re.search(r'Normalize\s*\(.*mean\s*=\s*(\[[^\]]+\]).*std\s*=\s*(\[[^\]]+\])', transform_str)
    if norm_match:
        mean_str = norm_match.group(1).strip()
        std_str = norm_match.group(2).strip()
        mean = ast.literal_eval(mean_str)
        std = ast.literal_eval(std_str)
        return mean, std
    return None, None

def deserialize_transform_fromstring(transform_str):
    resize_size = _parse_resize_param(transform_str)
    mean, std = _parse_normalize_params(transform_str)

    new_transform = T.Compose([
        T.Resize(resize_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return new_transform

def deserialize_transform(serialized_transform):
    if type(serialized_transform) == str:
        return deserialize_transform_fromstring(serialized_transform)
    out = []
    for step in serialized_transform:
        cls = getattr(T, step["type"])
        out.append(cls(**step["args"]))
    return T.Compose(out)

# ------------ .HDF5 dataset setup ------------------------

def create_domain_stratified_hdf5(hdf5_path, train_df, test_df, ordered_domains, img_path_cols, transforms, val_fraction=0.25, seed=42, compression="gzip"):
    """
    Create a single HDF5 file for all domains with domain-stratified train/val splits and test data.
    Each domain stores all image variants for all splits.
    Structure like:
    file.hdf5
    ├── Home/
    │   ├── train/
    │   │   ├── images/{image variants}
    │   │   │   ├── {variant 1}
    │   │   │   ...
    │   │   └── labels
    │   ├── val/
    │   │   ├── images/{image variants}
    │   │   │   ├── {variant 1}
    │   │   │   ...
    │   │   └── labels
    │   └── test/
    │       ├── images/{image variants}
    │       │   ├── {variant 1}
    │       │   ...
    │       └── labels
    ├── BigOffice-2/
    │   ├── ...
    ...
    """
    assert len(transforms) == len(img_path_cols), f"Assertion: Number of image path columns ({len(img_path_cols)}) has to equal number of transforms ({len(transforms)})."

    # Create reproducible splits and metadata
    domains = ordered_domains
    domain_to_index = {d: i for i, d in enumerate(domains)}

    with h5py.File(hdf5_path, 'w') as f:
        # Store domain indices
        for d, idx in domain_to_index.items():
            f.attrs[f'domain_index_{d}'] = idx

        # Process each domain separately
        for domain in domains:
            domain_group = f.require_group(domain)
            domain_df = train_df[train_df['domain'] == domain]

            # Split into train/val by image_path
            unique_paths = domain_df['image_path'].values
            train_paths, val_paths = train_test_split(unique_paths,
                                                      test_size=val_fraction,
                                                      random_state=seed)
            
            split_defs = {
                'train': domain_df[domain_df['image_path'].isin(train_paths)],
                'val': domain_df[domain_df['image_path'].isin(val_paths)]
            }

            # Optionally include test set for this domain
            test_domain_df = test_df[test_df['domain'] == domain]
            if not test_domain_df.empty:
                split_defs['test'] = test_domain_df

            for split_name, sdf in split_defs.items():
                split_grp = domain_group.require_group(split_name)
                # store labels
                scaled_labels = (sdf[LABEL_COLS].values.astype(np.float32) - 1.0) / 4.0
                split_grp.create_dataset('labels', data=scaled_labels, compression=compression)
                
                # create sub-group for image variants
                imgs_grp = split_grp.require_group('images')

                for img_col, transform in zip(img_path_cols, transforms):
                    imgs = []
                    for path in tqdm(sdf[img_col], desc=f"{domain}/{split_name}/{img_col}"):
                        with Image.open(path).convert('RGB') as img:
                            img = transform(img)
                            imgs.append(img.numpy())
                    imgs_np = np.stack(imgs)
                    imgs_grp.create_dataset(img_col, data=imgs_np, compression=compression)
    print(f"✅ HDF5 written at {hdf5_path}")
    return True

def add_image_variant_to_hdf5(
    hdf5_path,
    train_df,
    test_df,
    ordered_domains,
    new_img_col,
    new_transform,
    val_fraction=0.25,
    seed=42,
    compression="gzip"
):
    """
    Add a new image variant as a dataset across all domains/splits to an existing HDF5.
    - train_df, test_df: must have the new image column in the same order as original.
    - new_img_col: str, name of new column in the DataFrames and name for HDF5 dataset.
    - new_transform: your torchvision transform for this image view.
    """
    # Split train/val identically as before for reproducibility
    with h5py.File(hdf5_path, "a") as f:
        for domain in ordered_domains:
            domain_df = train_df[train_df['domain'] == domain]
            unique_paths = domain_df['image_path'].values
            train_paths, val_paths = train_test_split(
                unique_paths, test_size=val_fraction, random_state=seed
            )
            split_defs = {
                'train': domain_df[domain_df['image_path'].isin(train_paths)],
                'val': domain_df[domain_df['image_path'].isin(val_paths)]
            }
            test_domain_df = test_df[test_df['domain'] == domain]
            if not test_domain_df.empty:
                split_defs['test'] = test_domain_df

            for split_name, sdf in split_defs.items():
                imgs_grp = f[f"{domain}/{split_name}/images"]
                if new_img_col in imgs_grp:
                    print(f"Dataset already exists: {domain}/{split_name}/images/{new_img_col}")
                    continue

                imgs_new = []
                for path in tqdm(sdf[new_img_col], desc=f"Adding {domain}/{split_name}/{new_img_col}"):
                    with Image.open(path).convert('RGB') as img:
                        img = new_transform(img)
                        imgs_new.append(img.numpy())
                imgs_np = np.stack(imgs_new)
                imgs_grp.create_dataset(new_img_col, data=imgs_np, compression=compression)
    print(f"✅ Added {new_img_col} to all domains/splits in {hdf5_path}")

import h5py

def delete_variant_datasets(hdf5_path, variant_name, ordered_domains, splits=['train', 'val', 'test']):
    with h5py.File(hdf5_path, 'a') as f:
        for domain in ordered_domains:
            if domain not in f:
                continue
            for split in splits:
                group_path = f"{domain}/{split}/images"
                if group_path in f:
                    imgs_grp = f[group_path]
                    if variant_name in imgs_grp:
                        print(f"Deleting {group_path}/{variant_name}")
                        del imgs_grp[variant_name]
                    else:
                        print(f"{variant_name} not found in {group_path}")

    print(f"✅ Variant datasets '{variant_name}' deleted where present.")


class MultiVariantHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, domain, split,
                 img_path_cols, return_labels=True,
                 set_first_element_as_domain_label=False, transforms=[None,None]):
        self.hdf5_path = hdf5_path
        self.domain = domain
        self.split = split
        self.img_path_cols = img_path_cols
        self.transforms = transforms
        self.return_labels = return_labels
        self.set_first_element_as_domain_label = set_first_element_as_domain_label
        self._f = None  # lazy open per worker

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            return len(f[f'{self.domain}/{self.split}/labels'])

    def __getitem__(self, idx):
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, 'r')

        dom_split = f'{self.domain}/{self.split}'
        img_grp = self._f[f'{dom_split}/images']
        labels_ds = self._f[f'{dom_split}/labels']

        images = []
        for col, transform in zip(self.img_path_cols, self.transforms):
            img_tensor = torch.from_numpy(img_grp[col][idx]) 
            if transform is not None:
                img_tensor = transform(img_tensor)
            images.append(img_tensor)

        # domain index saved as file attribute; retrieve once lazily
        if not hasattr(self, 'domain_index'):
            self.domain_index = None
            for k, v in self._f.attrs.items():
                if k.endswith(f'_{self.domain}'):
                    self.domain_index = torch.tensor(v, dtype=torch.long)
                    break

        if self.return_labels:
            labels = torch.tensor(labels_ds[idx], dtype=torch.float32)
            domain_label = self.domain_index
            if self.set_first_element_as_domain_label:
                images.insert(0, domain_label)
            return tuple(images), labels, domain_label
        else:
            return tuple(images)


def get_domain_dataloaders_from_hdf5(hdf5_path, domains, img_path_cols, set_first_element_as_domain_label=False,
                                     batch_sizes=(32,64,64), num_workers=0,
                                     pin_memory=torch.cuda.is_available(), persistent_workers=False, transforms=[None, None]):
    persistent_workers = persistent_workers if num_workers > 0 else False
    domain_dataloaders = {}
    for domain in domains:
        loaders = {}
        for split, bs in zip(['train', 'val', 'test'], batch_sizes):
            ds = MultiVariantHDF5Dataset(hdf5_path, domain, split,
                                            img_path_cols,
                                            set_first_element_as_domain_label=set_first_element_as_domain_label, transforms=transforms)
            shuffle = (split == 'train')
            loaders[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        domain_dataloaders[domain] = loaders
    return domain_dataloaders


# ------------------- disk images + dataset setup ------------------------

class ImageDataset(Dataset):
    def __init__(self, df, transforms, img_path_cols=['image_path'], domain_col='domain', return_labels=True, set_first_element_as_domain_label=False):
        assert len(transforms) == len(img_path_cols), f"Assertion: Number of image path columns ({len(img_path_cols)}) has to equal number of transforms ({len(transforms)})."
        self.df = df.reset_index(drop=True)
        self.img_path_cols = img_path_cols
        self.domain_col = domain_col
        self.transforms = transforms
        self.category_indexes = {cat: idx for idx, cat in enumerate(df[domain_col].unique())}
        self.return_labels = return_labels
        self.set_first_element_as_domain_label = set_first_element_as_domain_label
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        images = []
        for img_path_col, transform in zip(self.img_path_cols, self.transforms):
            img_path = str(self.df.iloc[idx][img_path_col])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Error loading {img_path}: {str(e)}")
            
            img = transform(img) 
            images.append(img)
            
        raw_labels = self.df.iloc[idx][LABEL_COLS].values.astype(np.float32)
        scaled_labels = (raw_labels - 1) / 4
        scaled_labels = torch.from_numpy(scaled_labels)
        
        category_label = self.df.iloc[idx][self.domain_col]
        category_index = self.category_indexes[category_label]
        category_index = torch.tensor(category_index, dtype=torch.long)

        if self.set_first_element_as_domain_label:
            images.insert(0, category_index)

        if self.return_labels:
            return tuple(images), scaled_labels, category_index
        else:
            return tuple(images)


def _create_dataloaders(df, transforms, img_path_cols, include_test, set_first_element_as_domain_label=False, batch_sizes=(32, 64, 64), seed=42, num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False):
    """Create train/val dataloaders using image_path as unique key"""
    persistent_workers = persistent_workers if num_workers > 0 else False

    # Get image paths as indexing for split
    unique_images = df[['image_path']].reset_index(drop=True)
    
    # Split using image_path as key
    train_paths, val_paths = train_test_split(
        unique_images['image_path'], 
        test_size=0.25, 
        random_state=seed
    )
   
    # Create subsets
    train_df = df[df['image_path'].isin(train_paths)]
    val_df = df[df['image_path'].isin(val_paths)]
    
    train_dataset = ImageDataset(train_df, transforms=transforms, img_path_cols=img_path_cols, set_first_element_as_domain_label=set_first_element_as_domain_label)
    val_dataset = ImageDataset(val_df, transforms=transforms, img_path_cols=img_path_cols, set_first_element_as_domain_label=set_first_element_as_domain_label)
    
    # Create loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers),
        'val': DataLoader(val_dataset, batch_size=batch_sizes[1], shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    }
    
    if include_test is not False:
        test_dataset = ImageDataset(include_test, transforms=transforms, img_path_cols=img_path_cols, set_first_element_as_domain_label=set_first_element_as_domain_label)
        loaders['test']= DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,)

    split_idx = {'train': train_paths, 'val': val_paths}

    return (loaders, split_idx)


def get_domain_dataloaders(df, transforms, img_path_cols, include_test=False, set_first_element_as_domain_label=False, return_splits=False, batch_sizes=(32, 64, 64), seed=42, num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False):
    """
    Creates domain stratifed dataloaders

    """
    assert (include_test is False) or isinstance(include_test, pd.DataFrame), f"Assertion: include_test is {include_test}, should be pd.DataFrame."

    domain_dataloaders = {}
    domain_splits = {}
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        df_test = include_test[include_test['domain'] == domain] if include_test is not False else False
        loaders, split_idx = _create_dataloaders(domain_df, transforms, img_path_cols, include_test=df_test, set_first_element_as_domain_label=set_first_element_as_domain_label, batch_sizes=batch_sizes, seed=seed, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        domain_dataloaders[domain] = loaders
        domain_splits[domain] = split_idx

    return domain_dataloaders if not return_splits else (domain_dataloaders, domain_splits)

import warnings
def combine_all_dataloaders(domain_dataloaders):
    warnings.warn("`combine_all_dataloaders` is deprecated; use `pool_domain_dataloaders` instead", DeprecationWarning, stacklevel=2)
    pool_domain_dataloaders(domain_dataloaders)

def pool_domain_dataloaders(domain_dataloaders):

    train_datasets = [dl.dataset for dl in [domain_dataloaders[d]['train'] for d in domain_dataloaders]]
    val_datasets   = [dl.dataset for dl in [domain_dataloaders[d]['val'] for d in domain_dataloaders]]
    test_datasets  = [dl.dataset for dl in [domain_dataloaders[d]['test'] for d in domain_dataloaders]]


    all_train_dataset = ConcatDataset(train_datasets)
    all_val_dataset   = ConcatDataset(val_datasets)
    all_test_dataset  = ConcatDataset(test_datasets)

    exemplar_dataloader = domain_dataloaders.popitem()[1]
    all_train_loader = DataLoader(all_train_dataset, batch_size=exemplar_dataloader['train'].batch_size, shuffle=True)
    all_val_loader   = DataLoader(all_val_dataset, batch_size=exemplar_dataloader['val'].batch_size, shuffle=False)
    all_test_loader  = DataLoader(all_test_dataset, batch_size=exemplar_dataloader['test'].batch_size, shuffle=False)

    all_domain_dataloaders={}
    all_domain_dataloaders['joint']={
        'train':all_train_loader,
        'val':all_val_loader,
        'test':all_test_loader
    }
    return all_domain_dataloaders


def get_crossvalidation_domain_loaders(df, folds, transforms, img_path_cols, batch_sizes=(32, 64, 64), seed=42, num_workers=0, set_first_element_as_domain_label=False, pin_memory=torch.cuda.is_available()):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_loaders = {}

    # Loop over each fold
    for fold_num, (train_val_idx, test_idx) in enumerate(skf.split(df, df['domain'])):
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        domain_loaders = get_domain_dataloaders(train_val_df, transforms=transforms, img_path_cols=img_path_cols, include_test=test_df, batch_sizes=batch_sizes, seed=seed, num_workers=num_workers, set_first_element_as_domain_label=set_first_element_as_domain_label, pin_memory=pin_memory)
        fold_loaders[fold_num] = domain_loaders
        
    return fold_loaders


def stratified_train_test_split(dataframe_path='../data/pepper_data.pkl', stratify_by_col='domain', seed=42) -> tuple[str, str]:
    """
    Split every the data into train and test datasets (80:20 split).
    The split is applied to every domain separately and all splits are then combined.
    The splits are done using reset index as unique key.
    Saves train and test DataFrames as pickle files in the same directory, with suffixes '_train.pkl' and '_test.pkl'.

    Returns:
        Tuple of (train_path: str, test_path: str) where datasets are saved.
    """
    global_df = pd.read_pickle(dataframe_path).reset_index(drop=True)

    global_test_idx = []
    for category in global_df[stratify_by_col].unique():
        df = global_df[global_df[stratify_by_col] == category]

        unique_idx = df.index.tolist()

        train_idx, temp_idx = train_test_split(unique_idx, test_size=0.4, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
        global_test_idx.extend(test_idx)

    df_test = global_df.loc[global_test_idx]
    df_train = global_df.drop(global_test_idx)

    source_path = Path(dataframe_path)
    test_path = source_path.with_name(source_path.stem + "_test.pkl").as_posix()
    train_path = source_path.with_name(source_path.stem + "_train.pkl").as_posix()

    df_test.to_pickle(test_path)
    df_train.to_pickle(train_path)

    return train_path, test_path


def equalize_test_data(test_dataframe_path='../data/pepper_data_test.pkl', seed=42):
    """
    Equalise all domains to the smallest domain (samplewise) to avoid domain size bias during testing and save as new dataframe.
    """
    test_df = pd.read_pickle(test_dataframe_path)
    img_count = test_df.groupby("domain")['image_path'].nunique()
    smallest_domain_size = min(img_count)
    test_df_equal = test_df.groupby('domain', group_keys=False).sample(n=smallest_domain_size, random_state=seed)
    new_path = Path(test_dataframe_path).with_name(Path(test_dataframe_path).stem + '_equal.pkl')
    test_df_equal.to_pickle(new_path)
    return new_path