from ssc.data import read_curve, load_dataset, SSCStat
from ssc.processing import *
from ssc.simulation import simulate_real_apparent_dataset

def test_CutNegativeStrain() -> None:
    ssc = read_curve(data_path = "test_data/ssc_data_header.csv")
    ssc_cut = CutNegativeStrain()(input=ssc)
    assert ssc_cut.strain().gt(0).sum() == ssc_cut.length()

def test_RandomCut() -> None:
    ssc = read_curve(data_path = "test_data/ssc_data_header.csv")
    p_cut = 0.1
    ssc_cut = RandomCut(p = 1.0, p_cut = p_cut, p_start=1.0)(input=ssc)

    cut_size = int(ssc.length() * p_cut)
    assert ssc_cut.length() == ssc.length() - cut_size

def test_NormalizeSSC() -> None:
    dataset = load_dataset(root="test_data/test_dataset")
    stat = SSCStat(dataset)
    normalizer = NormalizeSSC(min_strain=stat.min_strain(),
                              max_strain=stat.max_strain(),
                              min_stress=stat.min_stress(),
                              max_stress=stat.max_stress())
    normalized = normalizer(input=dataset.data.to_list())

    assert len(normalized) == dataset.size
    for curve in normalized:
        assert curve.min_strain(strain_only = True) >= 0 and curve.max_strain(strain_only = True) <= 1
        assert curve.min_stress(stress_only = True) >= 0 and curve.max_stress(stress_only = True) <= 1

def test_SSC2Tensor() -> None:
    ssc = read_curve(data_path = "test_data/ssc_data_header.csv")
    t_ssc = SSC2Tensor()(input = ssc)

    assert t_ssc.shape == (ssc.length(), 2)

def test_XYRealApparentSplit() -> None:
    real_dataset = load_dataset("test_data/test_dataset")
    real_apparent_dataset = simulate_real_apparent_dataset(real_dataset)
    xy_split = XYRealApparentSplit()
    transformed = xy_split(real_apparent_dataset.data)
    for i, item in enumerate(transformed):
        X, Y, info = item

        assert X.shape == (real_apparent_dataset.data[i].length(), 2)
        assert Y.shape == (real_apparent_dataset.data[i].length(), )

def test_ProcessingPipeline() -> None:
    dataset = load_dataset(root="test_data/test_dataset")
    stat = SSCStat(dataset=dataset)
    pipeline = ProcessingPipeline([
        CutNegativeStrain(),
        RandomCut(p=0.5),
        NormalizeSSC(min_strain=stat.min_strain(),
                     max_strain=stat.max_strain(),
                     min_stress=stat.min_stress(),
                     max_stress=stat.max_stress())
    ])
    ssc = dataset.__getitem__([0,1])
    tensor = pipeline(ssc)

    assert isinstance(tensor, list)
