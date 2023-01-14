import random
import numpy as np
from pandas import Series

# -----------------
# --- CONSTANTS ---
# -----------------

STRAIN = "Strain"
STRESS = "Stress_MPa"
STRAIN_A = "Strain_a"

# -----------------
# --- FUNCTIONS --- 
# -----------------

def simulate_apparent_strain(strain_real: Series, r: float = None, alpha: float = 2) -> Series:
    """
        Simulate an apparent Strain-Stress curve given real strain values using the formula:

            strain_apparent = sqrt(strain_real * (1 + r)) / alpha

        Arguments:
        ----------

        strain_real : pd.Series
            Series of real strain values.

        r : float
            If `r = None`, a random value between 0 and 1 is used.

        alhpa : float
            Default = 2.0

        Return:
        -------

            Apparent strain values
    """
    if r == None:
        r = random.random()
        return np.sqrt(strain_real * (1 + r)) / alpha, r   
    return np.sqrt(strain_real * (1 + r)) / alpha
