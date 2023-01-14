import pandas as pd
import glob
from globals import *
import numpy as np

class StrainStressDataset():

    def __init__(self, 
                 root: str, 
                 r: float = None,
                 alpha: float = 2.0) -> None:
        """
            Arguments:
            ----------

            root : str
                Dataset root directory.

            r : float
                Real value used for apparent strain simulation.

                If `r = None` a random value between 0 am 1 is used
            
            alpha : float
                Real value used for apparent strain simulation.

                It must be > 0, otherwise it is set to 1.
        """
        self.root = root
        self.filenames = glob.glob(self.root + "/*.csv")
        self.size = len(self.filenames)
        self.r = r
        if self.r == None:
            self.r = random.random()
        self.alpha = alpha
        if self.alpha <= 0:
            self.alpha = 1

        self.load()
        self.calc_stat()

    def load(self):
        """
            Load dataset.
        """
        self.is_numpy = False
        self.data = {}
        for i, filename in enumerate(self.filenames):
            ssc = pd.read_csv(filename)
            ssc = ssc.where(ssc > 0).dropna(axis = 0)
            ssc[STRAIN_A] = simulate_apparent_strain(ssc[STRAIN], self.r, self.alpha)
            self.data.update({i: ssc})

    def to_numpy(self, min_max_norm: bool = False, standardize: bool = False, dtype = np.float32):
        """
            Convert this dataset in numpy form.

            Normilize or Standardize data if `min_max_morm` or `standardize` are respectively `True`.

            If both are `True`, data are Normalized.
        """
        self.is_numpy = True
        for key in self.data:
            ssc = self.data[key]
            if min_max_norm:
                ssc[STRAIN] = (ssc[STRAIN] - self.strain_min) / (self.strain_max - self.strain_min)
                ssc[STRESS] = (ssc[STRESS] - self.stress_min) / (self.stress_max - self.stress_min)
                ssc[STRAIN_A] = (ssc[STRAIN_A] - self.strain_min) / (self.strain_max - self.strain_min)
            elif standardize:
                ssc[STRAIN] = (ssc[STRAIN] - self.strain_mean) / self.strain_std
                ssc[STRESS] = (ssc[STRESS] - self.stress_mean) / self.stress_std
                ssc[STRAIN_A] = (ssc[STRAIN_A] - self.strain_mean) / self.strain_std

            self.data[key] = ssc.to_numpy(dtype = dtype)

    def calc_stat(self):
        """
            Compute the mean, standard dev., min and max values of the dataset.
        """
        if self.is_numpy:
            raise RuntimeError("Cannot compute statistics after numpy conversion.")
        
        strain_series = pd.Series(dtype=float)
        stress_series = pd.Series(dtype=float)
        strain_a_series = pd.Series(dtype=float)

        self.strain_min = None
        self.stress_min = None
        self.strain_a_min = None
        self.strain_max = None
        self.stress_max = None
        self.strain_a_max = None

        for i in range(self.size):
            ssc = self.data[i]
            strain_series = pd.concat([strain_series, ssc[STRAIN]])
            stress_series = pd.concat([stress_series, ssc[STRESS]])
            strain_a_series = pd.concat([strain_a_series, ssc[STRAIN_A]])

            sample_strain_min, sample_strain_max = strain_series.min(), strain_series.max()
            sample_stress_min, sample_stress_max = stress_series.min(), stress_series.max()
            sample_strain_a_min, sample_strain_a_max = strain_a_series.min(), strain_a_series.max()

            if self.strain_min == None or sample_strain_min < self.strain_min:
                self.strain_min = sample_strain_min
            if self.strain_max == None or sample_strain_max > self.strain_max:
                self.strain_max = sample_strain_max

            if self.stress_min == None or sample_stress_min < self.stress_min:
                self.stress_min = sample_stress_min
            if self.stress_max == None or sample_stress_max > self.stress_max:
                self.stress_max = sample_stress_max

            if self.strain_a_min == None or sample_strain_a_min < self.strain_a_min:
                self.strain_a_min = sample_strain_a_min
            if self.strain_a_max == None or sample_strain_a_max > self.strain_a_max:
                self.strain_a_max = sample_strain_a_max

        self.strain_mean, self.strain_std = strain_series.mean(), strain_series.std()
        self.stress_mean, self.stress_std = stress_series.mean(), stress_series.std()
        self.strain_a_mean, self.strain_a_std = strain_a_series.mean(), strain_a_series.std()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        strain_stress = self.data[idx]
        if self.is_numpy:
            return strain_stress[:, [2, 1]], strain_stress[:,0]
        return strain_stress.drop(STRAIN, axis = 1), strain_stress[STRAIN]