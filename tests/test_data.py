from ssc.data import read_curve, load_dataset

def test_read_ssc_with_header() -> None:
    STRAIN_LABEL = "Strain"
    STRESS_LABEL = "Stress_MPa"
    ssc = read_curve(data_path="test_data/ssc_data_header.csv",
                     header=0,
                     strain_idx=0,
                     stress_idx=1)
    assert ssc.curve.columns[0] == ssc.strain_label == STRAIN_LABEL
    assert ssc.curve.columns[1] == ssc.stress_label == STRESS_LABEL
    assert ssc.length() == 49

def test_read_ssc_with_header_no_indices() -> None:
    STRAIN_LABEL = "Strain"
    STRESS_LABEL = "Stress_MPa"
    ssc = read_curve(data_path="test_data/ssc_data_header.csv",
                     header=0,
                     strain_idx=None,
                     stress_idx=None,
                     strain_label=STRAIN_LABEL,
                     stress_label=STRESS_LABEL)
    assert ssc.curve.columns[0] == ssc.strain_label == STRAIN_LABEL
    assert ssc.curve.columns[1] == ssc.stress_label == STRESS_LABEL
    assert ssc.length() == 49

def test_read_ssc_no_header() -> None:
    STRAIN_LABEL = "strain"
    STRESS_LABEL = "stress"
    ssc = read_curve(data_path="test_data/ssc_data_no_header.csv",
                     header=None,
                     strain_idx=0,
                     stress_idx=1,
                     strain_label=STRAIN_LABEL,
                     stress_label=STRESS_LABEL)
    assert ssc.curve.columns[0] == ssc.strain_label == STRAIN_LABEL
    assert ssc.curve.columns[1] == ssc.stress_label == STRESS_LABEL
    assert ssc.length() == 49

def test_load_dataset() -> None:
    ssc_dataset = load_dataset(root = "test_data/test_dataset",
                                header=0,
                                strain_idx=0,
                                stress_idx=0)
    assert ssc_dataset.size == 3

def TODO_test_SSCDataLoader():
    # TODO: implementare
    raise NotImplementedError()
    