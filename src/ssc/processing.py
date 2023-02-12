"""
    Classes and functions for data processing.
"""

# --------------
# --- IMPORT ---
# --------------

import random
from pandas import DataFrame, Series
from ssc.data import StrainStressCurve, RealApparentSSC, SSCStat, RealApparentSSCStat
from torch import tensor
import numpy as np

# ---------------
# --- CLASSES ---
# ---------------

# TODO: check which processors return an object copy and which modify the object inplace. 

class BaseProcessor():
    """
        Base class for batch data processing.
    """
    def __init__(self) -> None:
        pass

    def process(self, object, index: int = None, batch_size : int = None):
        """
            Process a single object.

            Arguments:
            ----------

            object:
                Object to process.

            index:
                Object index. Default = `None`.

            batch_size:
                Size of data bacth the given object belongs to. Default = `None`
        """
        raise NotImplementedError()

    def __call__(self, input: 'object | list | Series') -> 'object | list':
        if isinstance(input, list) or isinstance(input, Series):
            batch_size = len(input)
            toReturn = []
            for i, obj in enumerate(input):
                toReturn.append(self.process(obj, i, batch_size))
            return toReturn
        return self.process(input)

class LambdaProcessor(BaseProcessor):

    def __init__(self, _lambda) -> None:
        super().__init__()
        self._lambda = _lambda

    def process(self, object, index: int = None, batch_size : int = None):
        return self._lambda(object, index, batch_size)

class CutNegativeStrain(BaseProcessor):
    """
        Cut all curve points before the last negative Strain value.
    """
    def __init__(self):
        super(CutNegativeStrain, self).__init__()

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        # Check negative values
        nv = object.strain().lt(0).sum()
        if nv > 0:
            # Compute the index of the last negative value
            # 1) get the stain series
            # 2) compute a boolean series of values < 0 
            # 3) reverse it
            # 4) call idxmax
            cut_idx = object.strain().lt(0).iloc[::-1].idxmax()
            cut = object.curve.iloc[cut_idx + 1:]
            cut.reset_index(drop=True, inplace=True)
            if isinstance(object, RealApparentSSC):
                return RealApparentSSC(curve = cut, like=object, id=object.id)
            return StrainStressCurve(curve = cut, like=object, id=object.id)
        return object

class CutFirstN(BaseProcessor):
    """
        Cut first n points.
    """
    def __init__(self, n : int) -> None:
        super().__init__()
        assert n > 0
        self.n = n

    def process(self, object, index: int = None, batch_size: int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        cut = object.curve.iloc[self.n:]
        cut.reset_index(drop=True, inplace=True)
        if isinstance(object, RealApparentSSC):
            return RealApparentSSC(curve=cut, like=object, id=object.id)
        return StrainStressCurve(curve=cut, like=object, id=object.id)

class CutLastN(BaseProcessor):
    """
        Cut last n points.
    """
    def __init__(self, n: int) -> None:
        super().__init__()
        assert n > 0
        self.n = n

    def process(self, object, index: int = None, batch_size: int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        cut = object.curve.iloc[:-self.n]
        cut.reset_index(drop=True, inplace=True)
        if isinstance(object, RealApparentSSC):
            return RealApparentSSC(curve=cut, like=object, id=object.id)
        return StrainStressCurve(curve=cut, like=object, id=object.id)

class RandomCut(BaseProcessor):
    """
        Randomly cut a Strain-Stress curve
    """
    def __init__(self,
                 p: float,
                 p_cut: float = 0.1,
                 p_start: float = 0.5) -> None:
        """
            Arguments:
            ----------

            p : float
                Probability of cut.

            p_cut : float
                Percentage of points to cut down.

            p_start : float
                Probability of cut side.

                With `p_start` probability the cut is done at the start of the curve,
                with `1 - p_start` probability is done at the end.
        """
        self.p = p
        self.p_cut = p_cut
        self.p_start = p_start

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        if random.random() >= self.p:
            return object
        curve_length = object.length()
        cut_size = int(curve_length * self.p_cut)
        if random.random() < self.p_start:
            return CutFirstN(cut_size).process(object, index, batch_size)
        return CutLastN(cut_size).process(object, index, batch_size)

class AddApparentNoise(BaseProcessor):
    """
    Add random noise to Apparent Strain values.

    Noise is sampled from a Normal Distribution.
    """
    def __init__(self,
                 noise_mean: float = 0.0,
                 noise_std: float = 0.01,
                 inplace: bool = True) -> None:
        """
        Arguments:
        ----------
            noise_mean : float
                Mean of additive noise.
            
            noise_std : float
                Standard deviation of additive noise.

            inplace : bool
                Whether to modify the original object or to return a copy.
        """
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.inplace = inplace

    def process(self, object, index: int = None, batch_size: int = None):
        if not isinstance(object, RealApparentSSC):
            raise TypeError("Invalid type: input must be a ApparentSSCSimulation object.")
        if not self.inplace:
            object = object.copy()
        curve_len = object.length()
        noise = np.random.normal(loc=self.noise_mean, scale=self.noise_std, size=(curve_len,))
        object.curve[object.apparent_strain_label] += noise

        return object

class NormalizeSSC(BaseProcessor):
    """
        Compute the Min-Max Normalization of a Strain-Stress curve.
    """
    def __init__(self, stats: SSCStat = None, use_apparent_values: bool = True) -> None:
        """
            Arguments:
            ----------

            stats : SSCStat
                Statistics of a Strain-Stress cuve dataset where minimum and maximum values are taken from.

                If `stat == None` each curve is normalized on its own min and max value.

            use_apparent_values : bool
                If `True`, the apparent curve is normalized with minimum and maximum 
                apparent strain values, otherwise real values are used.
        """
        self.stats = stats
        self.use_apparent_values = use_apparent_values
        if self.stats != None:
            if isinstance(self.stats, SSCStat):
                self.min_strain = self.stats.min_strain()
                self.max_strain = self.stats.max_strain()
                self.min_stress = self.stats.min_stress()
                self.max_stress = self.stats.max_stress()
            if isinstance(self.stats, RealApparentSSCStat) and self.use_apparent_values:
                self.min_app_strain = self.stats.min_apparent_strain()
                self.max_app_strain = self.stats.max_apparent_strain()
            else:
                self.min_app_strain = self.min_strain
                self.max_app_strain = self.max_strain

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        strain = object.strain()
        stress = object.stress()
        if self.stats == None:
            min_strain = strain.min()
            max_strain = strain.max()
            min_stress = stress.min()
            max_stress = stress.max()
        else:
            min_strain = self.min_strain
            max_strain = self.max_strain
            min_stress = self.min_stress
            max_stress = self.max_stress
        strain = (strain - min_strain) / (max_strain - min_strain)
        stress = (stress - min_stress) / (max_stress - min_stress)
        norm_curve = DataFrame(data = zip(strain, stress), columns = [object.strain_label, object.stress_label])
        if isinstance(object, RealApparentSSC):
            apparent_strain = object.apparent_strain()
            if self.stats == None:
                if self.use_apparent_values:
                    min_app_strain = apparent_strain.min()
                    max_app_strain = apparent_strain.max()
                else:
                    min_app_strain = min_strain
                    max_app_strain = max_strain
            else:
                min_app_strain = self.min_app_strain
                max_app_strain = self.max_app_strain
            apparent_strain = (apparent_strain - min_app_strain) / (max_app_strain - min_app_strain)
            norm_curve.loc[:, object.apparent_strain_label] = apparent_strain
            return RealApparentSSC(curve=norm_curve, like=object, id=object.id)
        return StrainStressCurve(curve=norm_curve, like=object, id=object.id)

class SSC2Tensor(BaseProcessor):
    """
        Convert a Strain-Stress curve into Pytorch Tensors.
    """
    def __init__(self, device: str = "cpu", dtype = None) -> None:
        self.device = device
        self.dtype = dtype

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, StrainStressCurve):
            raise TypeError("Invalid type: input must be a StrainStressCurve object.")
        return tensor(data=object.curve.values, device=self.device, dtype=self.dtype)

class XYRealApparentSplit(BaseProcessor):
    """
        Given a `RealApparentSSC` return a tuple `(X, Y, info)`,
        where `X` is the Apparent curve, `Y` is the real Strain values 
        and `info` is a dictionary with additional curve info.
        
        Both `X` and `Y` are tensor.
    """
    def __init__(self, device: str = "cpu", dtype = None) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.ssc2tensor = SSC2Tensor(device = self.device, dtype = self.dtype)

    def process(self, object, index: int = None, batch_size: int = None):
        if not isinstance(object, RealApparentSSC):
            raise TypeError("Invalid type: input must be a RealApparentSSC object.")
        X = self.ssc2tensor(object.apparent_ssc())
        Y = tensor(data=object.strain().values, device=self.device, dtype=self.dtype)
        return X, Y, object.info()

class ProcessingPipeline():
    """
        Apply a sequence of transformation to a given input.
    """
    def __init__(self, processors: 'list[BaseProcessor]') -> None:
        self.processors = processors

    def __call__(self, x):
        # TODO: gestire i dataset come input
        for processor in self.processors:
            x = processor(x)
        return x
