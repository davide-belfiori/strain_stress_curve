"""
    Classes and functions for data handling.
"""

# --------------
# --- IMPORT ---
# --------------

import pandas as pd
import os
import fnmatch
import random

# ---------------
# --- CLASSES ---
# ---------------

# >>> Base Objects

class StrainStressCurve():
    """
        Describe a simple Strain-Stress curve.
    """
    def __init__(self,
                 curve: pd.DataFrame,
                 strain_label: 'str | int' = "strain",
                 stress_label: 'str | int' = "stress",
                 like: 'StrainStressCurve' = None,
                 id : 'str | int' = None):
        """
            Initialize a `StrainStressCurve` given a Pandas Dataframe

            Arguments:
            ----------

            curve : Dataframe
                Pandas Dataframe containing Strain and Stress values.

            strain_label : str | int
                Label or index of to the Strain column.

            stress_label : str | int
                Label or index of the Stress column. 

            like : StrainStressCurve
                If `like != None` the Strain and Stress labels are copied from this object.

            id : str | int
                Identifier (optional)
        """
        assert curve.shape[1] == 2
        self.curve = curve
        if like != None:
            self.strain_label = like.strain_label
            self.stress_label = like.stress_label
        else:
            self.strain_label = strain_label
            self.stress_label = stress_label
        if not self.strain_label in self.curve.columns :
           raise ValueError("Invalid Strain label.")
        if not self.stress_label in self.curve.columns :
           raise ValueError("Invalid Stress label.")
        self.id = id

    def copy(self):
        """
            Return a deep copy of this object.
        """
        return StrainStressCurve(curve=self.curve.copy(),
                                 strain_label=self.strain_label,
                                 stress_label=self.stress_label,
                                 id=self.id)

    def info(self):
        """
            Return curve info.
        """
        return {"id": self.id,
                "min_strain": self.min_strain(),
                "max_strain": self.max_strain(),
                "min_stress": self.min_stress(),
                "strain_mean": self.strain_mean(),
                "stress_mean": self.stress_mean(),
                "strain_std": self.strain_std(),
                "stress_std": self.stress_std()}

    def strain(self):
        """
            Return the Strain values of the curve.
        """
        return self.curve.loc[:, self.strain_label]

    def stress(self):
        """
            Return the Stress values of the curve.
        """
        return self.curve.loc[:, self.stress_label]

    def max_stress(self, stress_only: bool = False):
        """
            Return the maximum Stress point.

            Argumrnts:
            ----------

            stress_only : bool
                If `True`, return only the Stress value.
        """
        idx_max = self.curve[self.stress_label].idxmax()
        if stress_only:
            return self.curve.iloc[idx_max][self.stress_label]
        return self.curve.iloc[idx_max]

    def min_stress(self, stress_only: bool = False):
        """
            Return the minimum Stress point.

            Argumrnts:
            ----------

            stress_only : bool
                If `True`, return only the Stress value.
        """
        idx_min = self.curve[self.stress_label].idxmin()
        if stress_only:
            return self.curve.iloc[idx_min][self.stress_label]
        return self.curve.iloc[idx_min]

    def max_strain(self, strain_only: bool = False):
        """
            Return the maximum Strain point.

            Argumrnts:
            ----------

            strain_only : bool
                If `True`, return only the Strain value.
        """
        idx_max = self.curve[self.strain_label].idxmax()
        if strain_only:
            return self.curve.iloc[idx_max][self.strain_label]
        return self.curve.iloc[idx_max]

    def min_strain(self, strain_only: bool = False):
        """
            Return the minimum Strain point.

            Argumrnts:
            ----------

            strain_only : bool
                If `True`, return only the Strain value.
        """
        idx_min = self.curve[self.strain_label].idxmin()
        if strain_only:
            return self.curve.iloc[idx_min][self.strain_label]
        return self.curve.iloc[idx_min]

    def strain_mean(self):
        """
            Return the mean Strain value.
        """
        return self.curve[self.strain_label].mean()

    def stress_mean(self):
        """
            Return the mean Stress value.
        """
        return self.curve[self.stress_label].mean()

    def strain_std(self):
        """
            Return the Strain standard deviation.
        """
        return self.curve[self.strain_label].std()

    def stress_std(self):
        """
            Return the Stress standard deviation.
        """
        return self.curve[self.stress_label].std()

    def slope_value(self):
        """
            Not Implemented.
        """
        raise NotImplementedError()

    def length(self):
        """
            Return the number of curve points.
        """
        return self.curve.shape[0]

    def __len__(self):
        return self.length()

    # TODO: implement __getitem__ method for indexing handling

class RealApparentSSC(StrainStressCurve):
    """
        Describe a cuple of Real an Apparent Strain-Stress curve.

        Params :
        --------

        curve : DataFrame
            Real Strain-Stress curve.

        app_strain : Series
            Apparent Strain values.
    """
    def __init__(self, 
                 curve: 'pd.DataFrame | tuple[pd.DataFrame, pd.Series]', 
                 strain_label: 'str | int' = "strain", 
                 stress_label: 'str | int' = "stress", 
                 apparent_strain_label: 'str | int' = "strain_apparent",
                 like: 'RealApparentSSC' = None,
                 id : 'str | int' = None):
        """
            Initialize a `RealApparentSSC` given a Pandas Dataframe

            Arguments:
            ----------

            curve : DataFrame | tuple[DataFrame, Series]
                Can be a Pandas Dataframe containing Strain, Stress and Apparent Strain values, or
                a tuple where the first element is a Strain-Stress Dataframe 
                and the second is a Series of Apparent Strain values.

            strain_label : str | int
                Label or index of to the Strain column.

            stress_label : str | int
                Label or index of the Stress column. 

            apparent_strain_label : str | int
                Label or index of the Apparent Strain column.

            like : RealApparentSSC
                If `like != None` the Strain, Stress and Apparent Strain labels are copied from this object.

            id : str | int
                Identifier (optional)
        """
        if like != None:
            self.strain_label = like.strain_label
            self.stress_label = like.stress_label
            self.apparent_strain_label = like.apparent_strain_label
        else:
            self.strain_label = strain_label
            self.stress_label = stress_label
            self.apparent_strain_label = apparent_strain_label

        if isinstance(curve, pd.DataFrame):
            assert curve.shape[1] == 3
            if not self.strain_label in curve.columns :
                raise ValueError("Invalid Strain label.")
            if not self.stress_label in curve.columns :
                raise ValueError("Invalid Stress label.")
            if not self.apparent_strain_label in curve.columns :
                raise ValueError("Invalid Apparent Strain label.")
            self.curve = curve

        elif isinstance(curve, tuple):
            assert len(curve) == 2
            assert isinstance(curve[0], pd.DataFrame) and curve[0].shape[1] == 2
            assert isinstance(curve[1], pd.Series) and curve[1].shape[0] == curve[0].shape[0]

            if not self.strain_label in curve[0].columns :
                raise ValueError("Invalid Strain label.")
            if not self.stress_label in curve[0].columns :
                raise ValueError("Invalid Stress label.")

            self.curve = pd.DataFrame(data=zip(curve[0][strain_label], curve[0][stress_label], curve[1]),
                                      columns=[self.strain_label, self.stress_label, self.apparent_strain_label])
        self.id = id

    def copy(self):
        return RealApparentSSC(data = self.curve.copy(),
                                strain_label = self.strain_label,
                                stress_label = self.stress_label,
                                apparent_strain_label = self.apparent_strain_label,
                                id=self.id)

    def info(self):
        i = super().info()
        i.update({"min_apparent_strain": self.max_apparent_strain(),
                  "max_apparent_strain": self.max_apparent_strain(),
                  "apparent_strain_mean": self.apparent_strain_mean(),
                  "apparent_strain_std": self.apparent_strain_std()})
        return i

    def real_ssc(self) -> StrainStressCurve:
        """
            Return the real Strain-Stress curve.
        """
        return StrainStressCurve(curve = self.curve.drop(self.apparent_strain_label, axis = 1),
                                 strain_label = self.strain_label,
                                 stress_label = self.stress_label)

    def apparent_ssc(self)-> StrainStressCurve:
        """
            Retutn the apparent Strain-Stress curve.
        """
        return StrainStressCurve(curve = self.curve.drop(self.strain_label, axis = 1),
                                 strain_label = self.apparent_strain_label,
                                 stress_label = self.stress_label)

    def apparent_strain(self):
        """
            Return the Apparent Strain vlaues.
        """
        return self.curve.loc[:, self.apparent_strain_label]

    def min_apparent_strain(self, app_strain_only: bool = False):
        """
            Return the minimum Apparent Strain point.

            Argumrnts:
            ----------

            app_strain_only : bool
                If `True`, return only the Apparent Strain value.
        """
        idx_min = self.curve[self.apparent_strain_label].idxmin()
        if app_strain_only:
            return self.curve.iloc[idx_min][self.apparent_strain_label]
        return self.curve.iloc[idx_min]

    def max_apparent_strain(self, app_strain_only: bool = False):
        """
            Return the maximum Apparent Strain point.

            Argumrnts:
            ----------

            app_strain_only : bool
                If `True`, return only the Apparent Strain value.
        """
        idx_max = self.curve[self.apparent_strain_label].idxmax()
        if app_strain_only:
            return self.curve.iloc[idx_max][self.apparent_strain_label]
        return self.curve.iloc[idx_max]

    def apparent_strain_mean(self):
        """
            Return the mean Apparent Strain value.
        """
        return self.curve[self.apparent_strain_label].mean()

    def apparent_strain_std(self):
        """
            Return the Apparent Strain standard deviation.
        """
        return self.curve[self.apparent_strain_label].std()

# >>> Dataset

class BaseDataset():
    """
        Base class for Dataset implementation.
    """
    def __init__(self, data: pd.Series) -> None:
        self.data = data
        self.size = data.size

    def sample(self):
        """
            Return a random sample.
        """
        idx = random.choice(self.data.index)
        return self.data[idx]

    def train_test_split(self, test_ratio: float = 0.1, shuffle: bool = True):
        """
            Split the dataset into train and test subsets.

            Arguments:
            ----------

            test_ratio : float
                Size of test dataset. Must be in [0, 1], default = 0.1

            shuffle : bool
                Shuffle dataset before splitting.
        """
        assert test_ratio <= 1, test_ratio >= 0.
        test_size = int(self.size * test_ratio)
        train_size = self.size - test_size

        indices = self.data.index.to_list()
        if shuffle:
            random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        return BaseDataset(self.data[train_indices]), BaseDataset(self.data[test_indices])

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

class StrainStressDataset(BaseDataset):
    """
        Collection of `StrainStressCurve` objects.
    """
    def __init__(self, data: 'pd.Series[StrainStressCurve]') -> None:
        """
            Arguments:
            ----------

            data: Series
                Series of `StrainStressCurve` objects
        """
        super(StrainStressDataset, self).__init__(data)

    def __getitem__(self, idx) -> StrainStressCurve:
        return super().__getitem__(idx)

class RealApparentSSCDataset(BaseDataset):
    """
        Collection of `RealApparentSSC` objects.
    """
    def __init__(self, data: 'pd.Series[RealApparentSSC]') -> None:
        """
            Arguments:
            ----------

            data: Series
                Series of `RealApparentSSC` objects
        """
        super(RealApparentSSCDataset, self).__init__(data)
    
    def __getitem__(self, idx) -> RealApparentSSC:
        return super().__getitem__(idx)

# >>> Statistics

class SSCStat():
    """
        Statistics of a Strain-Stress Dataset.
    """
    def __init__(self, dataset: "StrainStressDataset | pd.Series | list[StrainStressCurve]"):
        if isinstance(dataset, StrainStressDataset):
            index = dataset.data.index
        elif isinstance(dataset, pd.Series):
            index = dataset.index
            dataset = dataset.__iter__()
        elif isinstance(dataset, list):
            index = None
        else:
            raise TypeError("Invalid type for dataset.")
        data = []
        for sample in dataset:
            data.append({"min_strain": sample.min_strain(strain_only = True),
                         "max_strain": sample.max_strain(strain_only = True),
                         "min_stress": sample.min_stress(stress_only = True),
                         "max_stress": sample.max_stress(stress_only = True),
                         "strain_mean": sample.strain_mean(),
                         "stress_mean": sample.stress_mean(),})
        self.stat_df = pd.DataFrame(data = data, index=index)

    def min_strain(self):
        """
            Return the minimum Strain value in the dataset.
        """
        return self.stat_df["min_strain"].min()
    
    def max_strain(self):
        """
            Return the minimum Strain value in the dataset.
        """
        return self.stat_df["max_strain"].max()

    def min_stress(self):
        """
            Return the minimum Stress value in the dataset.
        """
        return self.stat_df["min_stress"].min()
    
    def max_stress(self):
        """
            Return the minimum Stress value in the dataset.
        """
        return self.stat_df["max_stress"].max()

    def strain_mean(self):
        """
            Return the mean Strain value of the dataset.
        """
        return self.stat_df["strain_mean"].mean()
    
    def stress_mean(self):
        """
            Return the mean Stress value of the dataset.
        """
        return self.stat_df["stress_mean"].mean()

class RealApparentSSCStat(SSCStat):
    """
        Statistics of a Real-Apparent Strain-Stress Dataset.
    """
    def __init__(self, dataset: "RealApparentSSCDataset | pd.Series | list[RealApparentSSC]"):
        if isinstance(dataset, RealApparentSSCDataset):
            index = dataset.data.index
        elif isinstance(dataset, pd.Series):
            index = dataset.index
            dataset = dataset.__iter__()
        elif isinstance(dataset, list):
            index = None
        else:
            raise TypeError("Invalid type for dataset.")
        data = []
        for sample in dataset:
            data.append({"min_strain": sample.min_strain(strain_only = True),
                         "max_strain": sample.max_strain(strain_only = True),
                         "min_stress": sample.min_stress(stress_only = True),
                         "max_stress": sample.max_stress(stress_only = True),
                         "min_apparent_strain": sample.min_apparent_strain(app_strain_only=True),
                         "max_apparent_strain": sample.max_apparent_strain(app_strain_only=True),
                         "strain_mean": sample.strain_mean(),
                         "stress_mean": sample.stress_mean(),
                         "apparent_strain_mean": sample.apparent_strain_mean()})
        self.stat_df = pd.DataFrame(data = data, index=index)

    def min_apparent_strain(self):
        """
            Return the minimum Apparent Strain value in the dataset.
        """
        return self.stat_df["min_apparent_strain"].min()

    def max_apparent_strain(self):
        """
            Return the maximum Apparent Strain value in the dataset.
        """
        return self.stat_df["max_apparent_strain"].max()

    def apparent_strain_mean(self):
        """
            Return the mean Apparent Strain value of the dataset.
        """
        return self.stat_df["apparent_strain_mean"].mean()

# >>> Data loader

class SSCDataLoader():
    """
        Data loader for Strain-Stress dataset.
    """
    def __init__(self,
                 data: BaseDataset,
                 batch_size : int = 1,
                 shuffle: bool = True,
                 processing = None) -> None:
        """
            Arguments:
            ----------

            data : BaseDataset
                Dataset.
            
            batch_size : int
                Size of a batch.

            shuffle : bool
                Shuffle data.

            processing : ProcessingPipeline
                Batch-wise processing pipeline.
        """
        self.data = data
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.processing = processing

        self.current_batch_idx = 0
        if self.data.size % self.batch_size == 0:
            self.num_batches = self.data.size // self.batch_size
        else:
            self.num_batches = (self.data.size // self.batch_size) + 1
        self.indices = self.data.data.index.to_list()

        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        item = self.data.__getitem__(idx=idx)
        if self.processing:
            return self.processing(item)
        return item

    def __next__(self):
        if self.current_batch_idx >= self.num_batches:
            self.reset()
            raise StopIteration
        batch_start = self.current_batch_idx * self.batch_size
        batch_end = (self.current_batch_idx + 1) * self.batch_size
        indices = self.indices[batch_start : batch_end]
        self.current_batch_idx += 1
        return self.__getitem__(indices)

    def reset(self):
        """
            Reset data loader. 
        """
        self.current_batch_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)

# -----------------
# --- FUNCTIONS ---
# -----------------

def read_curve(data_path: str,
               header: int = 0,
               strain_idx: int = 0,
               stress_idx: int = 1,
               strain_label: str = "strain",
               stress_label: str = "stress",
               sep: str = ",") -> StrainStressCurve:
    """
        Read a Strain-Stress curve from a CSV file.

        Arguments:
        ----------

        data_path: str
            Path of CSV file

        header : int
            Index of header row in the CSV file.
            
            If `header = None` or `header < 0`, `strain_label` and `stress_label` are used as column names.

        strain_idx : int
            Index of Strain column.

            It msut be `!= None` and `>= 0` if `header = None`.

        stress_idx : int
            Index of Stress column.

            It msut be `!= None` and `>= 0` if `header = None`.

        strain_label : str
            Label of to the Strain column. 

            If `header` and `strain_idx` are both `!= None` this field is ignored.

        stress_label : str
            Label of the Stress column. Ignored if `header != None`.

            If `header` and `stress_idx` are both `!= None` this field is ignored. 

        sep : str
            Separator.

        Return:
        -------
        A `StrainStressCurve` object.
    """
    if header != None and header >= 0:
        df = pd.read_csv(data_path, header = header, sep = sep)
        if strain_idx != None and strain_idx >= 0:
            strain_label = df.columns[strain_idx]
        else:
            if strain_label == None:
                raise ValueError("Strain label must be specified if Strain column index is unknown or < 0")
            if strain_label not in df.columns:
                raise ValueError("{} not found in column names.".format(strain_label))
        if stress_idx != None and stress_idx >= 0:
            stress_label = df.columns[stress_idx]
        else:
            if stress_label == None:
                raise ValueError("Stress label must be specified if Stress column index is unknown or < 0")
            if stress_label not in df.columns:
                raise ValueError("{} not found in column names.".format(stress_label))
    else:
        df = pd.read_csv(data_path, header=None, sep = sep)
        if strain_idx == None or strain_idx < 0:
            raise ValueError("Strain index must be an integer > 0 when header index is not given.")
        if strain_label == None:
            raise ValueError("Strain label must be specified when header index is not given.")
        if stress_idx == None or stress_idx < 0:
            raise ValueError("Stress index must be an integer > 0 when header index is not given.")
        if stress_label == None:
            raise ValueError("Stress label must be specified when header index is not given.")
        
        strain_colname = df.columns[strain_idx]
        stress_colname = df.columns[stress_idx]
        df.rename(columns = {strain_colname: strain_label,
                             stress_colname: stress_label}, 
                  inplace=True)

    return StrainStressCurve(curve=df, strain_label=strain_label, stress_label=stress_label)

def load_dataset(root: str,
                 header: int = 0,
                 strain_idx: int = 0,
                 stress_idx: int = 1,
                 strain_label: str = "strain",
                 stress_label: str = "stress",
                 sep: str = ",",
                 use_filename_as_key: bool = False,
                 use_key_as_id: bool = True) -> StrainStressDataset:
    """
        Load a Strain-Stress dataset reading all the CSV files in a root directory.
    
        Arguments:
        ----------

        root : str
            Root directory.

        header : int
            Index of header row in the CSV file.
            
            If `header = None` or `header < 0`, `strain_label` and `stress_label` are used as column names.

        strain_idx : int
            Index of Strain column.

            It msut be `!= None` and `>= 0` if `header = None`.

        stress_idx : int
            Index of Stress column.

            It msut be `!= None` and `>= 0` if `header = None`.

        strain_label : str
            Label of to the Strain column. 

            If `header` and `strain_idx` are both `!= None` this field is ignored.

        stress_label : str
            Label of the Stress column. Ignored if `header != None`.

            If `header` and `stress_idx` are both `!= None` this field is ignored. 

        sep : str
            Separator.

        use_filename_as_key : bool
            If `True` the name of each file is used as key in the data dictionary.

        use_key_as_id : bool
            Set the dataset key as identifier for each curve.

        Return:
        -------
        A `StrainStressDataset` object.
    """
    data = {}
    with os.scandir(root) as filenames:
        for i, filename in enumerate(filenames):
            if fnmatch.fnmatch(filename, "*.csv"):
                file_path = os.path.join(root, filename.name)
                ssc = read_curve(data_path = file_path,
                                 header = header,
                                 strain_idx = strain_idx,
                                 stress_idx = stress_idx,
                                 strain_label = strain_label,
                                 stress_label = stress_label,
                                 sep = sep)
                if use_filename_as_key:
                    key = filename.name
                else:
                    key = i
                if use_key_as_id:
                    ssc.id = key
                data.update({key: ssc})
            i += 1
    data = pd.Series(data, dtype=object)
    return StrainStressDataset(data = data)

def load_real_apparent_dataset(root: str) -> RealApparentSSCDataset:
    """
        Load a Real-Apparent Strain-Stress Dataset. (Not Implemented)
    """
    # TODO: implementare
    raise NotImplementedError()
