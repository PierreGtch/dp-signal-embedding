import pickle

import torch
from torch import nn
from einops.layers.torch import Rearrange
from sklearn.base import TransformerMixin
from skorch import NeuralNet
from skorch.utils import to_numpy
from huggingface_hub import hf_hub_download
# imoprt braindecode # must be installed to unpickle the model

from signal_embedding.models.base import ModelGetter


class FrozenNeuralNetTransformer(NeuralNet, TransformerMixin):
    def __init__(
            self,
            module,
            *args,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=nn.Sequential(),
            **kwargs
        )
        self.initialize()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        X = self.infer(X)
        return to_numpy(X)


class SkorchTransformer(ModelGetter):
    def __init__(
            self,
            hf_repo_id: str = 'PierreGtch/EEGNetv4',
            hf_model_name: str = 'EEGNetv4_Lee2019_MI',
            device='cpu',
    ):
        self.hf_repo_id = hf_repo_id
        self.hf_model_name = hf_model_name
        self.device = device

    def __call__(
            self,
            sfreq: float,
            input_window_seconds: float,
            chs_info: list[dict],
    ):
        # soft dependencies

        # load and check model kwargs:
        path_kwargs = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=f'{self.hf_model_name}/kwargs.pkl',
        )
        with open(path_kwargs, 'rb') as f:
            kwargs = pickle.load(f)
        module_cls = kwargs['module_cls']
        module_kwargs = kwargs['module_kwargs']

        def check_arg(arg_name, val):
            if arg_name in module_kwargs:
                assert module_kwargs[arg_name] == val

        check_arg("n_times", int(input_window_seconds * sfreq) + 1)
        check_arg("input_window_samples", int(input_window_seconds * sfreq) + 1)  # old name
        check_arg('input_window_seconds', input_window_seconds)
        check_arg('sfreq', sfreq)
        check_arg('n_chans', len(chs_info))
        check_arg('in_chans', len(chs_info))  # old name
        if 'chs_info' in module_kwargs:
            assert len(module_kwargs['chs_info']) == len(chs_info)
            assert all(cha['ch_name'] == chb['ch_name'] for cha, chb in
                       zip(module_kwargs['chs_info'], chs_info))

        # load pre-trained weights:
        path_params = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=f'{self.hf_model_name}/model-params.pkl',
        )
        torch_module = module_cls(**module_kwargs)
        torch_module.eval()
        torch_module.load_state_dict(torch.load(path_params, map_location='cpu'))

        # remove clf layer and flatten the output:
        torch_module.final_layer = Rearrange('batch x y z -> batch (x z y)')

        # wrap model in a sklearn-compatible transformer:
        transformer = FrozenNeuralNetTransformer(
            module=torch_module,
            device=self.device,
        )
        return transformer
