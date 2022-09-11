import os
import torch
import pathlib
import numpy as np

from solution import Instance
from meta.classes import Job
from ml.dp.model.model import DpModel, LinearOnly
from targets.estimators.normalizer_config import NORMALIZERS

if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


class Oracle:

    def tardy_probability(self, state, jobs: [Job]):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def get_name(self):
        raise NotImplementedError("This method must be implemented by a subclass.")


class DummyOracle(Oracle):

    def tardy_probability(self, state, jobs):
        return 0.5 * np.ones(len(jobs))

    def get_name(self):
        return "DummyOracle"


# TODO
class EDDOracle(Oracle):

    def tardy_probability(self, state, jobs):
        # Pro ten job co je nejvic EDD vrati vic, ne z tech, ktery jsou fixovany
        pass

    def get_name(self):
        return "EDDOracle"


class PerfectOracle(Oracle):

    def tardy_probability(self, state, jobs: {int: Job}):
        return self.probs

    def get_name(self):
        return "PerfectOracle"


class MLOracle(Oracle):

    def __init__(self, model_path: str = "ml/dp/trained/0412_142108.pth"):
        self.model_path = model_path
        loaded = torch.load(model_path)
        self.model = DpModel(**loaded['config'].config['arch']['args'], input_dimension=24)
        self.model.load_state_dict(loaded['state_dict'])
        self.model.eval()
        self.normalizer = NORMALIZERS[loaded['config'].config['data_loader']['args']['normalizer_name']]

    def calculate_probs(self, inst: Instance):
        normalized = self.normalizer.feature_normalizer.normalize_x(inst)
        with torch.no_grad():
            out = self.model(normalized[None, :])
            self.probs = out

    def tardy_probability(self, state, jobs: {int: Job}):
        return self.probs

    def get_name(self):
        return "MLOracle" + self.model_path


class MLOracleStatic(Oracle):

    def __init__(self, model_path: str = "ml/dp/trained/cur_experiment/best.pth"):
        self.model_path = model_path
        loaded = torch.load(model_path)
        in_dim = 3 if 'Identity' in loaded['config'].config['data_loader']['args']['normalizer_name'] else 26
        if loaded['config'].config['arch']['type'] == 'LinearOnly':
            self.model = LinearOnly(**loaded['config'].config['arch']['args'], input_dimension=in_dim)
        else:
            self.model = DpModel(**loaded['config'].config['arch']['args'], input_dimension=in_dim)
        self.model.load_state_dict(loaded['state_dict'])
        self.model.eval()
        self.normalizer = NORMALIZERS[loaded['config'].config['data_loader']['args']['normalizer_name']]
        self.probs = None

    def calculate_probs(self, inst: Instance):
        normalized = self.normalizer.feature_normalizer.normalize_x(inst)
        with torch.no_grad():
            out = self.model(normalized[None, :])
            self.probs = out

    def tardy_probability(self, state, jobs: {int: Job}):
        return self.probs

    def get_name(self):
        return "MLOracleStatic" + self.model_path

