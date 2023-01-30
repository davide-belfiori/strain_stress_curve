"""
    Set of utility functions.
"""

# --------------
# --- IMPORT ---
# --------------

from ssc.data import StrainStressCurve, RealApparentSSC
from ssc.simulation import ApparentSSCSimulation
import matplotlib.pyplot as plt

# -----------------
# --- FUNCTIONS ---
# -----------------

# >>> Plot functions

def plot_ssc(ssc: StrainStressCurve, print_id: bool = True) -> None:
    """
        Plot a Strain-Stress curve.
    """
    title = "Strain-Stress Curve"
    if ssc.id != None and print_id:
        title = title + " (id = {})".format(ssc.id)
    plt.plot(ssc.strain(), ssc.stress())
    plt.title(label=title)
    plt.xlabel(xlabel="Strain")
    plt.ylabel(ylabel="Stress")
    plt.show()

def plot_ra_ssc(ra_ssc: RealApparentSSC, plot_epsilon: bool = True, print_id: bool = True) -> None:
    """
        Plot a Real-Aparent Strain-Stress curve.
    """
    title = "Real-Apparent Strain-Stress Curve"
    if ra_ssc.id != None and print_id:
        title = title + " (id = {})".format(ra_ssc.id)
    if plot_epsilon:
        plt.subplot(1, 2, 1, title=title, xlabel="Strain", ylabel="Stress")
        plt.plot(ra_ssc.strain(), ra_ssc.stress(), label="real")
        plt.plot(ra_ssc.apparent_strain(), ra_ssc.stress(), label="apparent")
        plt.legend(loc="lower right")
        plt.subplot(1, 2, 2, title="Epsilon", xlabel="apparent", ylabel="real")
        plt.plot(ra_ssc.apparent_strain(), ra_ssc.strain())
    else: 
        plt.plot(ra_ssc.strain(), ra_ssc.stress(), label="real")
        plt.plot(ra_ssc.apparent_strain(), ra_ssc.stress(), label="apparent")
        plt.legend(loc="lower right")
        plt.title(label=title)
        plt.xlabel(xlabel="Strain")
        plt.ylabel(ylabel="Stress")
    plt.show()
    
def plot_simulation(sim: ApparentSSCSimulation, plot_epsilon: bool = True, print_id: bool = True) -> None:
    """
        Plot a Real-Aparent Strain-Stress curve simulation.
    """
    title = "Real-Apparent Simulation"
    if sim.id != None and print_id:
        title = title + " (id = {})".format(sim.id)
    if plot_epsilon:
        plt.subplot(1, 2, 1, title=title, xlabel="Strain", ylabel="Stress")
        plt.plot(sim.strain(), sim.stress(), label="real")
        plt.plot(sim.apparent_strain(), sim.stress(), label="apparent (r = {:.4f})".format(sim.r))
        plt.legend(loc="lower right")
        plt.subplot(1, 2, 2, title="Epsilon", xlabel="apparent", ylabel="real")
        plt.plot(sim.apparent_strain(), sim.strain())
    else: 
        plt.plot(sim.strain(), sim.stress(), label="real")
        plt.plot(sim.apparent_strain(), sim.stress(), label="apparent")
        plt.legend(loc="lower right")
        plt.title(label=title)
        plt.xlabel(xlabel="Strain")
        plt.ylabel(ylabel="Stress")
    plt.show()
    