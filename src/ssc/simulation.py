"""
    Apparent Strain-Stress curve simulation.
"""

# --------------
# --- IMPORT ---
# --------------

from ssc.data import RealApparentSSC, StrainStressDataset, RealApparentSSCDataset, RealApparentSSCStat
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

    def info(self):
        i = super().info()
        i.update({"r": self.r,
                  "alpha": self.alpha})
        return i

class SimulateApparent(BaseProcessor):
    """
        Simulate an apparent Strain-Stress curve given the real one using the formula:

            strain_apparent = sqrt(strain_real * (1 + r)) / alpha
    """
    def __init__(self, 
                 r: float = None,
                 alpha: float = None, 
                 r_policy: 'Callable[[StrainStressCurve, int, int], float]' = None,
                 alpha_policy: 'Callable[[StrainStressCurve, int, int], float]' = None,
                 copy_id: bool = True) -> None:
        """
        Arguments:
        ----------

        real: StrainStressCurve
            Real Strain-Stress curve

        r: float
            If `r` and `r_policy` are `None`, a random value between 0 and 1 is used.

        alhpa : float
            If `alpha` and `alpha_policy` are `None`, 
            a random value is sampled from a normal distribution 
            with mean = 2and standard deviation = 0.1.

        r_policy : Callable[[StrainStressCurve, int, int], float]
            Function returning `r` value given a StrainStressCurve, 
            its position in the batch and the batch size.
        
        alpha_policy : Callable[[StrainStressCurve, int, int], float]
            Function returning `alpha` value given a StrainStressCurve, 
            its position in the batch and the batch size.

        copy_id : bool
            If `True`, simulation id is taken from the real `StrainStressCurve`
        """
        super(SimulateApparent, self)
        self.r = r
        self.alpha = alpha
        self.r_policy = r_policy
        self.alpha_policy = alpha_policy
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
        if self.alpha_policy == None:
            if self.alpha == None:
                alpha = random.normalvariate(mu=2, sigma=0.1)
            else:
                alpha = self.alpha
        else:
            alpha = self.alpha_policy(object, index, batch_size)
        apparent_strain = np.sqrt(strain_real * (1 + r)) / alpha
        return ApparentSSCSimulation(real = object,
                                     apparent_strain = apparent_strain,
                                     r = r,
                                     alpha = alpha,
                                     id = object.id if self.copy_id else None)

class CutSimulation(BaseProcessor):
    """
        Cut n points of a Real-Apparent simulation.
    """
    def __init__(self, n: int, from_end: bool = False) -> None:
        """
            Arguments:
            ----------

            n : int
                Number of points to cut.

            from_end : bool
                Cut last n points. (Default = False)
        """
        assert n > 0
        self.n = n
        self.from_end = from_end

        if self.from_end:
            self.ssc_cut = CutLastN(self.n)
        else:
            self.ssc_cut = CutFirstN(self.n)

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, ApparentSSCSimulation):
            raise TypeError("Invalid type: input must be a ApparentSSCSimulation object.")
        
        real_cut = self.ssc_cut(object.real_ssc())
        if self.from_end:
            apparent_cut = object.apparent_strain().iloc[:-self.n]
        else:
            apparent_cut = object.apparent_strain().iloc[self.n:]
        apparent_cut.reset_index(drop=True, inplace=True)

        return ApparentSSCSimulation(real = real_cut, 
                                     apparent_strain = apparent_cut,
                                     r = object.r,
                                     alpha = object.alpha,
                                     apparent_label = object.apparent_strain_label,
                                     id = object.id)

class NormalizeSimulation(NormalizeSSC):
    """
        Compute the Min-Max Normalization of a Real-Apparent Simulation.
    """
    def __init__(self, stats: RealApparentSSCStat = None, use_apparent_values: bool = True) -> None:
        """
            Arguments:
            ----------

            stats : RealApparentSSCStat
                Statistics of a Real-Apparent Strain-Stress cuve dataset where minimum and maximum values are taken from.

                If `stat == None` each curve is normalized on its own min and max value.

            use_apparent_values : bool
                If `True`, the apparent curve is normalized with minimum and maximum 
                apparent strain values, otherwise real values are used.
        """
        super().__init__(stats=stats, use_apparent_values=use_apparent_values)

    def process(self, object, index: int = None, batch_size : int = None):
        if not isinstance(object, ApparentSSCSimulation):
            raise TypeError("Invalid type: input must be a ApparentSSCSimulation object.")
        norm = super().process(object=object, index=index, batch_size=batch_size)
            
        return ApparentSSCSimulation(real = norm.real_ssc(), 
                                     apparent_strain = norm.apparent_strain(), 
                                     r = object.r,
                                     alpha = object.alpha,
                                     apparent_label = object.apparent_strain_label,
                                     id = object.id)

class RealApparentSimulationDataset(RealApparentSSCDataset):
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

    def __getitem__(self, idx) -> ApparentSSCSimulation:
        return super().__getitem__(idx)

# -----------------
# --- FUNCTIONS ---
# -----------------

# TODO: aggiungere documentazione
def simulate_real_apparent_dataset(dataset: StrainStressDataset, 
                                   r: float = 0.5, 
                                   alpha: float = 2.0, 
                                   r_policy: 'Callable[[StrainStressCurve, int, int], float]' = None,
                                   alpha_policy: 'Callable[[StrainStressCurve, int, int], float]' = None):
    """
        Simulate a Real-Apparent Dataset from a `StrainStressDataset`.
    """
    pipeline = ProcessingPipeline([
        DropNan(),
        ShiftToPositive(),
        SimulateApparent(r = r, 
                         alpha = alpha, 
                         r_policy = r_policy, 
                         alpha_policy = alpha_policy) # TODO: aggiungere possibilità di modificare la "apparent_strain_label"
    ])
    sim_data = pipeline(dataset.data)
    sim_data = Series(data = sim_data, index = dataset.data.index)
    return RealApparentSimulationDataset(data = sim_data)
