"""
    Apparent Strain-Stress curve simulation.
"""

# --------------
# --- IMPORT ---
# --------------

from ssc.data import RealApparentSSC, StrainStressDataset, BaseDataset
from ssc.processing import *
import random
import numpy as np
from pandas import Series

from typing import Callable

# ---------------
# --- CLASSES ---
# ---------------

class ApparentSSCSimulation(RealApparentSSC):
    """
        Result of an Apparent Strain-Stress curve simulation.    
    """
    def __init__(self,    
                 real: StrainStressCurve,
                 apparent_strain: Series,
                 r: float,
                 alpha: float,
                 apparent_label: str = None,
                 id : 'str | int' = None) -> None:
        """
            Arguments:
            ----------

            real : StrainStressCurve
                Real Strain-Stress curve.

            apparent_strain:
                Apparent Strain values.

            r: float
                First parameter of simulation.

            alpha: float
                Second parameter of simulation.
            
            apparent_label : str
                Label of the apparent Strain values.

                If `apparent_label = None`, the real Strain label will be used, 
                followed by the suffix "_apparent".
            
            id : str | int
                Identifier (optional)
        """
        assert real.length() == apparent_strain.size

        if apparent_label == None:
            apparent_label = str(real.strain_label) + "_apparent"
        data = real.curve.copy()
        data.loc[:,apparent_label] = apparent_strain
        super(ApparentSSCSimulation, self).__init__(curve = data,
                                                    strain_label=real.strain_label, 
                                                    stress_label=real.stress_label,
                                                    apparent_strain_label=apparent_label,
                                                    id=id)
        self.r = r
        self.alpha = alpha

    def copy(self):
        return ApparentSSCSimulation(real = self.real_ssc().copy(),
                                     apparent_strain = self.apparent_strain().copy(),
                                     r = self.r,
                                     alpha = self.alpha,
                                     apparent_label = self.apparent_strain_label)

class SimulateApparent(BaseProcessor):
    """
        Simulate an apparent Strain-Stress curve given the real one using the formula:

            strain_apparent = sqrt(strain_real * (1 + r)) / alpha
    """
    def __init__(self, 
                 r: float = None,
                 alpha: float = 2.0, 
                 r_policy: 'Callable[[StrainStressCurve, int, int], float]' = None,
                 copy_id: bool = True) -> None:
        """
        Arguments:
        ----------

        real: StrainStressCurve
            Real Strain-Stress curve

        r: float
            If `r` and `r_policy` are `None`, a random value between 0 and 1 is used.

        alhpa : float
            Default = 2.0

        r_policy : Callable[[StrainStressCurve, int, int], float]
            Function returning `r` value given a StrainStressCurve, 
            its position in the batch and the batch size.
        
        copy_id : bool
            If `True`, simulation id is taken from the real `StrainStressCurve`
        """
        super(SimulateApparent, self)
        self.r = r
        self.alpha = alpha
        self.r_policy = r_policy
        self.copy_id = copy_id

    def process(self, object: StrainStressCurve, index: int = None, batch_size: int = None) -> ApparentSSCSimulation:
        strain_real = object.strain()
        if self.r_policy == None:
            if self.r == None:
                r = random.random()
            else:
                r = self.r
        else:
            r = self.r_policy(object, index, batch_size)
        apparent_strain = np.sqrt(strain_real * (1 + r)) / self.alpha
        return ApparentSSCSimulation(real = object,
                                     apparent_strain = apparent_strain,
                                     r = r,
                                     alpha = self.alpha,
                                     id = object.id if self.copy_id else None)

# TODO: implementare una versione di XYSimSplit per RealApparentSSC
class XYSimSplit(XYRealApparentSplit):
    """
        Given an `ApparentSSCSimulation` return a tuple `(X, Y, r, alpha)`,
        where `X` is the Apparent curve and `Y` is the real Strain values.
        Both `X` and `Y` are tensor.
    """
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.ssc2tensor = SSC2Tensor(device = self.device)

    def process(self, object, index: int = None, batch_size: int = None):
        X, Y = super().process(object=object, index=index, batch_size=batch_size)
        return X, Y, object.r, object.alpha

class RealApparentSimulationDataset(BaseDataset):
    """
        Collection of `ApparentSSCSimulation` objects.
    """
    def __init__(self, data: 'Series[ApparentSSCSimulation]') -> None:
        """
            Arguments:
            ----------

            data: Series
                Series of `ApparentSSCSimulation` objects
        """
        super(RealApparentSimulationDataset, self).__init__(data=data)

# -----------------
# --- FUNCTIONS ---
# -----------------

# TODO: aggiungere documentazione
def simulate_real_apparent_dataset(dataset: StrainStressDataset, 
                                   r: float = 0.5, 
                                   alpha: float = 2.0, 
                                   r_policy: 'Callable[[StrainStressCurve, int, int], float]' = None):
    """
        Simulate a Real-Apparent Dataset from a `StrainStressDataset`.
    """
    pipeline = ProcessingPipeline([
        CutNegativeStrain(),
        SimulateApparent(r = r, alpha = alpha, r_policy = r_policy)
    ])
    sim_data = pipeline(dataset.data)
    sim_data = Series(data = sim_data, index = dataset.data.index)
    return RealApparentSimulationDataset(data = sim_data)
