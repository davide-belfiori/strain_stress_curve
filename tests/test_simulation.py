from ssc.data import read_curve, load_dataset
from ssc.processing import CutNegativeStrain
from ssc.simulation import *

def test_SimulateApparent() -> None:
    real_curve = read_curve(data_path = "test_data/ssc_data_header.csv")
    real_cut = CutNegativeStrain()(input = real_curve)
    res = SimulateApparent(r=0.5, alpha=2.0)(input=real_cut)
    assert res.length() == real_cut.length()

def test_simulate_real_apparent_dataset() -> None:
    real_dataset = load_dataset("test_data/test_dataset")
    real_apparent_dataset = simulate_real_apparent_dataset(real_dataset, 0.5, 2)

    assert real_apparent_dataset.size == real_apparent_dataset.size

def test_r_policy() -> None:
    real_dataset = load_dataset("test_data/test_dataset")
    r_policy = lambda ssc, index, batch_size: 1 if index % 2 == 0 else 0.5
    real_apparent_dataset = simulate_real_apparent_dataset(real_dataset, r_policy = r_policy)

    assert real_apparent_dataset.data[0].r == 1
    assert real_apparent_dataset.data[1].r == 0.5
    assert real_apparent_dataset.data[2].r == 1

def test_alpha_policy() -> None:
    real_dataset = load_dataset("test_data/test_dataset")
    alpha_policy = lambda ssc, index, batch_size: 1 if index % 2 == 0 else 2
    real_apparent_dataset = simulate_real_apparent_dataset(real_dataset, alpha_policy = alpha_policy)

    assert real_apparent_dataset.data[0].alpha == 1
    assert real_apparent_dataset.data[1].alpha == 2
    assert real_apparent_dataset.data[2].alpha == 1