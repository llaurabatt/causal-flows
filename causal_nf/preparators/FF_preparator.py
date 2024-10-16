from causal_nf.datasets.FF_dataset import FFDataset
from causal_nf.preparators.tabular_preparator import TabularPreparator
from causal_nf.sem_equations import sem_dict
from causal_nf.utils.io import dict_to_cn

from causal_nf.utils.scalers import StandardTransform

import networkx as nx
from torch.distributions import Normal, Independent
import torch


class FFPreparator(TabularPreparator):
    def __init__(self, add_noise, dataset_filename, **kwargs):

        self.dataset = None
        self.dataset_filename = dataset_filename
        self.add_noise = add_noise
        sem_fn = sem_dict["FF_data"](sem_name="dummy")

        self.adjacency = sem_fn.adjacency

        self.num_nodes = len(sem_fn.functions)

        self.intervention_index_list = sem_fn.intervention_index_list()
        super().__init__(name="FF_data", task="modeling", **kwargs)

        assert self.split == [0.8, 0.1, 0.1]

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = dict_to_cn(dataset)

        my_dict = {
            "add_noise": dataset.add_noise,
            "dataset_filename": dataset.dataset_filename,
        }

        my_dict.update(TabularPreparator.params(dataset))

        return my_dict

    @classmethod
    def loader(cls, dataset):
        my_dict = FFPreparator.params(dataset)

        return cls(**my_dict)

    def _x_dim(self):
        return self.num_nodes

    def get_intervention_list(self):
        raise NotImplementedError("FFPreparator.get_intervention_list")

    def diameter(self):
        adjacency = self.adjacency(True).numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.Graph)
        diameter = nx.diameter(G)
        return diameter

    def longest_path_length(self):
        adjacency = self.adjacency(False).numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
        longest_path_length = nx.algorithms.dag.dag_longest_path_length(G)
        return int(longest_path_length)

    def get_ate_list(self):
        raise NotImplementedError("FFPreparator.get_ate_list")

    def get_ate_list_2(self):
        raise NotImplementedError("FFPreparator.get_ate_list_2")

    def intervene(self, index, value, shape):
        # # one can only intervene on train data?
        # x = self.get_features_train()
        #
        # # shape never gets used??
        # if len(shape) == 1:
        #     shape = (shape[0], x.shape[1]) #
        #
        #
        # cond = x[..., index].floor() == int(value)
        # x = x[cond, :]
        #
        # return x[: shape[0]]
        raise NotImplementedError("FFPreparator.intervene")

    def compute_ate(self, index, a, b, num_samples=10000):
        # bug? why is this here?
        # ate = torch.rand((6)) * 2 - 1.0
        # return ate
        raise NotImplementedError("FFPreparator.compute_ate")

    def compute_counterfactual(self, x_factual, index, value):
        #
        # x_cf = torch.randn_like(x_factual)
        # x_cf[:, index] = value
        #
        # return x_cf
        raise NotImplementedError("FFPreparator.compute_counterfactual")

    def log_prob(self, x):
        px = Independent(
            Normal(
                torch.zeros(6),
                torch.ones(6),
            ),
            1,
        )
        return px.log_prob(x)

    def _loss(self, loss):
        if loss in ["default", "forward"]:
            return "forward"
        else:
            raise NotImplementedError(f"Wrong loss {loss}")

    def _split_dataset(self, dataset_raw):
        datasets = []

        for i, split_s in enumerate(self.split):
            dataset = FFDataset(
                root_dir=self.root, 
                split=self.split_names[i], 
                seed=self.k_fold,
                dataset_filename=self.dataset_filename,
            )

            dataset.prepare_data()
            dataset.set_add_noise(self.add_noise)
            if i == 0:
                self.dataset = dataset
            datasets.append(dataset)

        return datasets

    def _get_dataset(self, num_samples, split_name):
        raise NotImplementedError

    def get_scaler(self, fit=True):
        scaler = self._get_scaler()
        self.scaler_transform = None
        if fit:
            x = self.get_features_train()
            scaler.fit(x, dims=self.dims_scaler)
            if self.scale in ["default", "std"]:
                self.scaler_transform = StandardTransform(
                    shift=x.mean(0), scale=x.std(0)
                )
                print("scaler_transform", self.scaler_transform)

        self.scaler = scaler
        return scaler

    def get_scaler_info(self):
        if self.scale in ["default", "std"]:
            return [("std", None)]
        else:
            raise NotImplementedError

    @property
    def dims_scaler(self):
        return (0,)

    def _get_dataset_raw(self):
        return None

    def _transform_dataset_pre_split(self, dataset_raw):
        return dataset_raw

    def post_process(self, x, inplace=False):
        if not inplace:
            x = x.clone()
        dims = self.dataset.binary_dims
        min_values = self.dataset.binary_min_values
        max_values = self.dataset.binary_max_values

        x[..., dims] = x[..., dims].floor().float()
        x[..., dims] = torch.clamp(x[..., dims], min=min_values, max=max_values)

        return x

    def feature_names(self, latex=False):
        return self.dataset.column_names


    def _plot_data(
        self,
        batch=None,
        title_elem_idx=None,
        batch_size=None,
        df=None,
        hue=None,
        **kwargs,
    ):

        title = ""
        return super()._plot_data(
            batch=batch,
            title_elem_idx=title_elem_idx,
            batch_size=batch_size,
            df=df,
            title=title,
            hue=hue,
            diag_plot="hist",
        )
