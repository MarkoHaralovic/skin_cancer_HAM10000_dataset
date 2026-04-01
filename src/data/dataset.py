import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset — 7 diagnostic classes.

    Class indices:
        0  akiec  Actinic Keratoses / Bowen's Disease
        1  bcc    Basal Cell Carcinoma
        2  bkl    Benign Keratosis-like Lesions
        3  df     Dermatofibroma
        4  mel    Melanoma
        5  nv     Melanocytic Nevi  
        6  vasc   Vascular Lesions

    Usage
    -----
    Train / validation 
        train_ds = HAM10000Dataset(root, split='train', transform=train_tf)
        val_ds   = HAM10000Dataset(root, split='val',   transform=val_tf)
        # split is stratified per class — each class keeps the same ratio in train & val

    Test  (separate CSV + image folder):
        test_ds = HAM10000Dataset(
            root,
            metadata_csv='dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv',
            image_dirs=['dataverse_files/ISIC2018_Task3_Test_Images/ISIC2018_Task3_Test_Images'],
            split='test',
            transform=val_tf,
        )
        
        train and val split per row + group (lesion_id) aware
    """

    CLASS_NAMES    = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    MALIGNANT_CLASSES = {'mel', 'bcc', 'akiec'}
    _CLASS_TO_IDX  = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        root,
        metadata_csv='HAM10000_metadata',
        split='train',       # 'train' | 'val' | 'test'
        val_split=0.2,
        random_state=42,
        transform=None,
        image_dirs=None,
    ):
        self.transform = transform

        meta_path = metadata_csv if os.path.isabs(metadata_csv) \
                    else os.path.join(root, metadata_csv)
        meta = pd.read_csv(meta_path)

        if image_dirs is None:
            image_dirs = [
                os.path.join(root, d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        id_to_path = {}
        for folder in image_dirs:
            for fname in os.listdir(folder):
                if fname.lower().endswith('.jpg'):
                    id_to_path[os.path.splitext(fname)[0]] = os.path.join(folder, fname)

        meta['label'] = meta['dx'].map(self._CLASS_TO_IDX)
        meta = meta[meta['image_id'].isin(id_to_path)].copy()
        meta['path'] = meta['image_id'].map(id_to_path)

        if split == 'test':
            split_meta = meta.reset_index(drop=True)
        else:
            if 'lesion_id' not in meta.columns:
                raise ValueError("Grouped train/val split requires a 'lesion_id' column in metadata.")

            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=val_split, 
                random_state=random_state,
            )
            train_idx, val_idx = next(gss.split(meta, groups=meta['lesion_id']))
            train_meta = meta.iloc[train_idx]
            val_meta = meta.iloc[val_idx]
            
            split_meta = (train_meta if split == 'train' else val_meta).reset_index(drop=True)

        self.samples = list(zip(split_meta['path'], split_meta['label']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]
