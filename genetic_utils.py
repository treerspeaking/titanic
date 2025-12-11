import pandas as pd
import numpy as np

def GeneticFunction(data):
    res = ((np.minimum( ((((0.058823499828577 + data["Sex"]) - np.cos((data["Pclass"] / 2.0))) * 2.0)),  ((0.89))) * 2.0))
    res += np.maximum( ((data["SibSp"] - 2.409090042114258)),  ( -(np.minimum( (data["Sex"]),  (np.sin(data["Parch"]))) * data["Pclass"])))
    res += (0.14 * ((np.minimum( (data["Sex"]),  (((data["Parch"] / 2.0) / 2.0))) * data["Age"]) - data["Cabin"]))
    res += np.minimum( ((np.sin((data["Parch"] * ((data["Fare"] - 0.720430016517639) * 2.0))) * 2.0)),  ((data["SibSp"] / 2.0)))
    res += np.maximum( (np.minimum( ( -np.cos(data["Embarked"])),  (0.138462007045746))),  (np.sin(((data["Cabin"] - data["Fare"]) * 2.0))))
    res += -np.minimum( ((((data["Age"] * data["Parch"]) * data["Embarked"]) + data["Parch"])),  (np.sin(data["Pclass"])))
    res += np.minimum( (data["Sex"]),  ((np.sin( -(data["Fare"] * np.cos((data["Fare"] * 1.630429983139038)))) / 2.0)))
    res += np.minimum( ((0.25)),  (np.sin(np.minimum( (((67.0 / 2.0) * np.sin(data["Fare"]))),  (0.31830988618379069)))))
    res += np.sin((np.sin(data["Cabin"]) * (np.sin((12.6275)) * np.maximum( (data["Age"]),  (data["Fare"])))))
    res += np.sin(((np.minimum( (data["Fare"]),  ((data["Cabin"] * data["Embarked"]))) / 2.0) *  -data["Fare"]))
    res += np.minimum( (((2.675679922103882 * data["SibSp"]) * np.sin(((96) * np.sin(data["Cabin"]))))),  (data["Parch"]))
    res += np.sin(np.sin((np.maximum( (np.minimum( (data["Age"]),  (data["Cabin"]))),  ((data["Fare"] * 0.31830988618379069))) * data["Cabin"])))
    res += np.maximum( (np.sin(((12.4148) * (data["Age"] / 2.0)))),  (np.sin((-3.0 * data["Cabin"]))))
    res += (np.minimum( (np.sin((((np.sin(((data["Fare"] * 2.0) * 2.0)) * 2.0) * 2.0))), ), (data["SibSp"]))[0] / 2.0)
    res += ((data["Sex"] - data["SibSp"]) * (np.cos(((data["Embarked"] - 0.730768978595734) + data["Age"])) / 2.0))
    res += ((np.sin(data["Cabin"]) / 2.0) - (np.cos(np.minimum( (data["Age"]),  (data["Embarked"]))) * np.sin(data["Embarked"])))
    res += np.minimum( (0.33),  ((data["Sex"] * (2.212120056152344 * (0.720430016517639 - np.sin((data["Age"] * 2.0)))))))
    res += (np.minimum( (np.cos(data["Fare"])),  (np.maximum( (np.sin(data["Age"])),  (data["Parch"])))) * np.cos((data["Fare"] / 2.0)))
    res += np.sin((data["Parch"] * np.minimum( ((data["Age"] - 1.5707963267948966)),  ((np.cos((data["Pclass"] * 2.0)) / 2.0)))))
    res += (data["Parch"] * (np.sin(((data["Fare"] * (0.623655974864960 * data["Age"])) * 2.0)) / 2.0))
    res += (0.31830988618379069 * np.cos(np.maximum( ((0.602940976619720 * data["Fare"])),  ((np.sin(0.720430016517639) * data["Age"])))))
    res += (np.minimum( ((data["SibSp"] / 2.0)),  (np.sin(((data["Pclass"] - data["Fare"]) * data["SibSp"])))) * data["SibSp"])
    res += np.tanh((data["Sex"] * np.sin((5.199999809265137 * np.sin((data["Cabin"] * np.cos(data["Fare"])))))))
    res += (np.minimum( (data["Parch"]),  (data["Sex"])) * np.cos(np.maximum( ((np.cos(data["Parch"]) + data["Age"])),  (3.1415926535897931))))
    res += (np.minimum( (np.tanh(((data["Cabin"] / 2.0) + data["Parch"]))),  ((data["Sex"] + np.cos(data["Age"])))) / 2.0)
    res += (np.sin((np.sin(data["Sex"]) * (np.sin((data["Age"] * data["Pclass"])) * data["Pclass"]))) / 2.0)
    res += (data["Sex"] * (np.cos(((data["Sex"] + data["Fare"]) * ((8.48635) * (63)))) / 2.0))
    res += np.minimum( (data["Sex"]),  ((np.cos((data["Age"] * np.tanh(np.sin(np.cos(data["Fare"]))))) / 2.0)))
    res += (np.tanh(np.tanh( -np.cos((np.maximum( (np.cos(data["Fare"])),  (0.094339601695538)) * data["Age"])))) / 2.0)
    res += (np.tanh(np.cos((np.cos(data["Age"]) + (data["Age"] + np.minimum( (data["Fare"]),  (data["Age"])))))) / 2.0)
    res += (np.tanh(np.cos((data["Age"] * ((-2.0 + np.sin(data["SibSp"])) + data["Fare"])))) / 2.0)
    res += (np.minimum( (((281) - data["Fare"])),  (np.sin((np.maximum( ((176)),  (data["Fare"])) * data["SibSp"])))) * 2.0)
    res += np.sin(((np.maximum( (data["Embarked"]),  (data["Age"])) * 2.0) * (((785) * 3.1415926535897931) * data["Age"])))
    res += np.minimum( (data["Sex"]),  (np.sin( -(np.minimum( ((data["Cabin"] / 2.0)),  (data["SibSp"])) * (data["Fare"] / 2.0)))))
    res += np.sin(np.sin((data["Cabin"] * (data["Embarked"] + (np.tanh( -data["Age"]) + data["Fare"])))))
    res += (np.cos(np.cos(data["Fare"])) * (np.sin((data["Embarked"] - ((734) * data["Fare"]))) / 2.0))
    res += ((np.minimum( (data["SibSp"]),  (np.cos(data["Fare"]))) * np.cos(data["SibSp"])) * np.sin((data["Age"] / 2.0)))
    res += (np.sin((np.sin((data["SibSp"] * np.cos((data["Fare"] * 2.0)))) + (data["Cabin"] * 2.0))) / 2.0)
    res += (((data["Sex"] * data["SibSp"]) * np.sin(np.sin( -(data["Fare"] * data["Cabin"])))) * 2.0)
    res += (np.sin((data["SibSp"] * ((((5.428569793701172 + 67.0) * 2.0) / 2.0) * data["Age"]))) / 2.0)
    res += (data["Pclass"] * (np.sin(((data["Embarked"] * data["Cabin"]) * (data["Age"] - (1.07241)))) / 2.0))
    res += (np.cos((((( -data["SibSp"] + data["Age"]) + data["Parch"]) * data["Embarked"]) / 2.0)) / 2.0)
    res += (0.31830988618379069 * np.sin(((data["Age"] * ((data["Embarked"] * np.sin(data["Fare"])) * 2.0)) * 2.0)))
    res += ((np.minimum( ((data["Age"] * 0.058823499828577)),  (data["Sex"])) - 0.63661977236758138) * np.tanh(np.sin(data["Pclass"])))
    res += -np.minimum( ((np.cos(((727) * ((data["Fare"] + data["Parch"]) * 2.0))) / 2.0)),  (data["Fare"]))
    res += (np.minimum( (np.cos(data["Fare"])),  (data["SibSp"])) * np.minimum( (np.sin(data["Parch"])),  (np.cos((data["Embarked"] * 2.0)))))
    res += (np.minimum( (((data["Fare"] / 2.0) - 2.675679922103882)),  (0.138462007045746)) * np.sin((1.5707963267948966 * data["Age"])))
    res += np.minimum( ((0.0821533)),  (((np.sin(data["Fare"]) + data["Embarked"]) - np.cos((data["Age"] * (9.89287))))))
    return res

def get_genetic_feature(df):
    data = df.copy()
    # Sex
    data['Sex'] = data['Sex'].fillna('0')
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    # Cabin
    data['Cabin'] = data['Cabin'].fillna('0')
    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
    cabin_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, '0':0}
    data['Cabin'] = data['Cabin'].map(cabin_map).fillna(0)

    # Embarked
    emb_map = {'C':1, 'Q':2, 'S':3}
    data['Embarked'] = data['Embarked'].map(emb_map).fillna(0)

    # Fill remaining NaNs with -1 (Age, Fare, SibSp, Parch)
    features_needed = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Age', 'Cabin', 'Fare', 'Embarked']
    data = data[features_needed].fillna(-1).astype(float)

    return GeneticFunction(data)
