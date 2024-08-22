import pandas as pd
from torch.utils.data import Dataset
import os
import torch


def compute_split_idx(original_len, split_sizes, k_fold=None):
    all_idx = torch.arange(original_len)
    if len(split_sizes) == 1:
        return [all_idx]
    if isinstance(k_fold, int):
        assert k_fold >= 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        perm = torch.randperm(original_len, generator=generator)
        all_idx = all_idx[perm]
        n = len(perm) * (1 - split_sizes[0])
        all_idx = torch.roll(all_idx, shifts=int(n * k_fold))
    start_idx, end_idx = 0, None
    all_idx_splits = []

    num_splits = len(split_sizes)
    for i, size in enumerate(split_sizes):
        assert isinstance(size, float)
        assert 0 < size
        assert 1 > size
        new_len = int(size * original_len)
        end_idx = new_len + start_idx
        if i == (num_splits - 1):
            all_idx_splits.append(all_idx[start_idx:])
        else:
            all_idx_splits.append(all_idx[start_idx:end_idx])
        start_idx = end_idx

    return all_idx_splits


# %%
class FFDataset(Dataset):
    def __init__(self, root_dir: str, 
                 split: str, 
                 dataset_filename: str, 
                 seed: int = None):
        self.root_dir = root_dir
        self.dataset_filename = dataset_filename

        self.seed = seed
        self.split = split

        self.column_names = [
            "Z1",
            "Z2",
            "Z3",
            "Z4",
            "X",
            "Y",
        ]
        self.binary_dims = [2,3,4]
        self.binary_min_values = torch.tensor([0.0,0.0,0.0])
        self.binary_max_values = torch.tensor([1.0,1.0,1.0])
        self.x = None
        self.y = None
        self._add_noise = False

    def set_add_noise(self, value):
        self._add_noise = value

    def _create_data(self):
        """
        This method sets the value for self.X and self.U
        Returns: None

        """
        # data_df = pd.read_csv(os.path.join(self.root_dir, "ate_1_version0.csv"))[self.column_names]
        data_df = pd.read_csv(os.path.join(self.root_dir, self.dataset_filename))[self.column_names] 
        data = torch.from_numpy(data_df.values).float()

        num_samples = data.shape[0]

        all_idx_splits = compute_split_idx(
            original_len=num_samples, split_sizes=[0.8, 0.1, 0.1], k_fold=self.seed
        )

        if self.split == "train":
            data = data[all_idx_splits[0]]
        elif self.split in ["valid", "val"]:
            data = data[all_idx_splits[1]]
        elif self.split == "test":
            data = data[all_idx_splits[2]]
        else:
            raise NotImplementedError(f"Split {self.split} not implemented")

        y = torch.from_numpy(data_df['Y'].values).float()  # data[:, -1]
        x = data
        return x, y

    def prepare_data(self) -> None:
        print("\nPreparing data...")
        x, y = self._create_data()

        self.x = x
        self.y = y

    def data(self, one_hot=False, scaler=None, x=None):
        if x is not None:
            x_tmp = x.clone()
        else:
            x_tmp = self.x.clone()

        if scaler:
            x = scaler.transform(x_tmp)
        else:
            x = x_tmp
        return x, self.y

    def __getitem__(self, index):
        x = self.x[index].clone()
        return x, self.y[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        my_str = "Dataset FF\n"
        my_str += f"\tcolumns: {self.column_names}\n"
        return my_str
