import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn_extra import GroupImputer

train_df = pd.read_csv("./data/train.csv")

pipeline = make_pipeline(
    # (GroupImputer(["Pclass", "Sex"], "Age")),
    # (GroupImputer(["Pclass", "Sex"], "Fare")),
    (GroupImputer(["Ticket"], "Cabin", strategy="most_common")),
    # (FunctionTransformer(add_family_size, validate=False)),
    # (FunctionTransformer(strip_name, validate=False)),
    # (FunctionTransformer(extract_title, validate=False)),
    # (FunctionTransformer(extract_deck, validate=False)),
)

pipeline.fit(train_df)

new_df = pipeline.transform(train_df)