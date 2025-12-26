from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin

class GroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 group_cols,
                 target_col,
                 strategy: Literal["median", "mean", "most_common"]="median"
    ):
        """
        Args:
            group_cols (list): List of columns to group by (e.g., ['Pclass', 'Sex']).
            target (str): The column to impute (e.g., 'Age').
            metric (str): The aggregation metric ('mean' or 'median').
        """
        self.group_cols = group_cols
        self.target_col = target_col
        self.strategy = strategy
        self.impute_map_ = None
        self.global_fill_ = None

    def fit(self, X, y=None):
        # Calculate the metric for the target column grouped by the group_cols
        if self.strategy == "median":
            self.impute_map_ = X.groupby(self.group_cols)[self.target_col].median()
            self.global_fill_ = X[self.target_col].median()
        elif self.strategy == "mean":
            self.impute_map_ = X.groupby(self.group_cols)[self.target_col].mean()
            self.global_fill_ = X[self.target_col].mean()
        elif self.strategy == "most_common":
            # keep only the most common categorical
            assert X[self.target_col].dtype.name in ['object', 'category']
            # group by default drop all keys row that contain NA
            self.impute_map_ = X \
                                    .groupby(self.group_cols + [self.target_col])\
                                    .size() \
                                    .reset_index() \
                                    .sort_values(by=0, ascending=False) \
                                    .drop_duplicates(subset=self.group_cols, keep="first") \
                                    .drop(columns=0) \
                                    .set_index(self.group_cols)[self.target_col]
                                    # turn this into a series as everyone other implementation is also Series
            # for Missings
            self.global_fill_ = "M"
        return self

    def transform(self, X):
        # Create a copy to avoid SettingWithCopy warnings
        X = X.copy()

        # 1. Map the learned values to the rows based on group_cols
        # We index X by the group columns to align with impute_map_ index
        base_index = X.index
        X_indexed = X.set_index(self.group_cols)

        # map_values will contain the mean/median for the specific group of that row
        map_values = self.impute_map_.reindex(X_indexed.index)

        # Reset index alignment to original X
        map_values.index = base_index

        # 2. Fill missing values in target
        X[self.target_col] = X[self.target_col].fillna(map_values)

        # 3. Fallback: Fill any remaining NaNs (e.g., unseen groups) with global average
        X[self.target_col] = X[self.target_col].fillna(self.global_fill_)

        return X

    def set_output(self, *, transform = None):
        return self