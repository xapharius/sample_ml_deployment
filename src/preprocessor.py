import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, RobustScaler

def _cabin_floor(df):
    return df["cabin"].str[0].fillna("")


def _names(df):
    return df["name"].str.split(",").apply(lambda x: x[0])


def _fare(df):
    return np.log10(df["fare"] + 1)


class Preprocessor(object):
    
    def __init__(self):
        self.cabin_floor = LabelEncoder()
        self.sex = LabelEncoder()
        self.embarked = LabelBinarizer()
        self.name_feats = None
        self.scaler = RobustScaler()
    
    def _fit(self, df: pd.DataFrame):
        self.cabin_floor = self.cabin_floor.fit(_cabin_floor(df))
        self.sex = self.sex.fit(df["sex"])
        
        names = _names(df)
        self.name_feats_gb = _fare(df).groupby(names).mean().to_frame() 
        self.name_feats_gb["count"] = np.log(df.groupby(names).size())
        self.name_feats_gb = self.name_feats_gb.reset_index()
        
        self.embarked = LabelBinarizer().fit(df["embarked"].fillna("X"))
        
        return self
    
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        res["cabin_floor"] = np.log(self.cabin_floor.transform(_cabin_floor(df)) + 1)
        res["pclass"] = df["pclass"]
        res["sex"] = self.sex.transform(df["sex"])
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
        
        cols = ["embarked_" + str(i) for i in range(4)]
        embarked = pd.DataFrame(self.embarked.transform(df["embarked"].fillna("X")), 
                                columns=cols, index=df.index)
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
        
        