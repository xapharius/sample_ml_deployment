import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, RobustScaler


def _cabin_floor(df):
    res = df["cabin"].str[0].fillna("X")
    res = res.replace({"A": 9, "B": 7, "C": 6, "D": 5, 
                       "E": 4, "F": 3, "G": 2, "T": 1, "X": 0})
    return np.log(res + 1)


def _names(df):
    return df["name"].str.split(",").apply(lambda x: x[0])


def _fare(df):
    return np.log10(df["fare"] + 1)


def _sex(df):
    return df["sex"].replace({"male": 1, "female": 0})


def _embarked(df):
    res = pd.DataFrame()
    res["embarked_C"] = df["embarked"] == "C"
    res["embarked_Q"] = df["embarked"] == "Q"
    res["embarked_S"] = df["embarked"] == "S"
    res["embarked_X"] = df["embarked"].isnull()
    return res


class Preprocessor(object):
    
    def __init__(self):
        self.name_feats = None
        self.scaler = RobustScaler()
    
    def _fit(self, df: pd.DataFrame):
        names = _names(df)
        self.name_feats_gb = _fare(df).groupby(names).mean().to_frame() 
        self.name_feats_gb["count"] = np.log(df.groupby(names).size())
        self.name_feats_gb = self.name_feats_gb.reset_index()
        return self
    
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        res["cabin_floor"] = _cabin_floor(df)
        res["pclass"] = df["pclass"]
        res["sex"] = _sex(df)
        res["age"] = df["age"]
        res["fare"] = _fare(df)
        res[["sibsp", "parch"]] = np.log(df[["sibsp", "parch"]] + 1)
        res["missing_age"] = res["age"].isnull().astype(int)
        res["age"] = res["age"].fillna(0)
        
        name_feats = pd.merge(_names(df).to_frame(), self.name_feats_gb, 
                              left_on="name", right_on="name", how="left")
        name_feats = name_feats.set_index(df.index)
        res["name_mean_fare"] = name_feats["fare"].fillna(0)
        res["name_count"] = name_feats["count"].fillna(0)
        
        embarked = _embarked(df)
        res = pd.concat([res, embarked], axis=1)
        return res
    
    def _label_preproc(member):
        """
        Closure to deal with missing survivial column.
        """
        def wrapper(self, df):
            survived = pd.DataFrame()
            if "survived" in df.columns:
                survived = df[["survived"]]
                df = df.drop("survived", axis=1)
            res = member(self, df)
            if isinstance(res, pd.DataFrame):
                res = pd.concat([res, survived], axis=1)
            return res

        return wrapper

    @_label_preproc
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = self._transform(df)
        res = pd.DataFrame(self.scaler.transform(res), columns=res.columns, index=df.index)
        return res
    
    @_label_preproc
    def fit(self, df: pd.DataFrame):
        self._fit(df)
        res = self._transform(df)
        self.scaler.fit(res)
        return self
    
    @_label_preproc
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._fit(df)
        res = self._transform(df)
        res = pd.DataFrame(self.scaler.fit_transform(res), columns=res.columns, index=df.index)
        return res
        
        