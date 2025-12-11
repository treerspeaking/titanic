import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')

def load_data():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    gt_path = './out/gt.csv'
    gt_df = None
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)

    return train_df, test_df, gt_df

def get_family_survival_feature(df):
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Family_Survival'] = 0.5

    for grp, grp_df in df.groupby(['Surname', 'Fare']):
        if (len(grp_df) > 1):
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

    for _, grp_df in df.groupby('Ticket'):
        if (len(grp_df) > 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) or (row['Family_Survival'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin == 0.0):
                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

    return df

def preprocess(train_df, test_df):
    ntrain = train_df.shape[0]
    test_df_temp = test_df.copy()
    test_df_temp['Survived'] = np.nan
    all_data = pd.concat([train_df, test_df_temp]).reset_index(drop=True)

    # 1. Family Survival
    all_data = get_family_survival_feature(all_data)

    # 2. Standard Preprocessing
    all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
    all_data['Fare'] = all_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

    all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
    all_data['Title'] = all_data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)

    all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    all_data['Age'] = all_data.groupby(['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())

    # KEEPING CONTINUOUS VARS + BINS
    all_data['AgeBin'] = pd.cut(all_data['Age'], 5, labels=False)
    all_data['FareBin'] = pd.qcut(all_data['Fare'], 5, labels=False)

    # Deck
    all_data['Deck'] = all_data['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    all_data['Deck'] = all_data['Deck'].replace(['A', 'B', 'C'], 'ABC')
    all_data['Deck'] = all_data['Deck'].replace(['D', 'E'], 'DE')
    all_data['Deck'] = all_data['Deck'].replace(['F', 'G'], 'FG')
    all_data['Deck'] = LabelEncoder().fit_transform(all_data['Deck'])

    all_data['Embarked'] = all_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Interactions
    all_data['Age*Class'] = all_data['Age'] * all_data['Pclass']

    # Drop
    all_data = all_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Surname'], axis=1)
    # Keeping Age and Fare this time!

    X_train = all_data[:ntrain].drop('Survived', axis=1)
    X_test = all_data[ntrain:].drop('Survived', axis=1)

    return X_train, X_test

def main():
    train_df, test_df, gt_df = load_data()

    X_train, X_test = preprocess(train_df, test_df)
    y_train = train_df['Survived']

    # Models
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    gbc = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )

    ada = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    # Voting
    voting = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('gbc', gbc), ('ada', ada)],
        voting='soft'
    )

    voting.fit(X_train, y_train)

    predictions = voting.predict(X_test)

    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'")

    if gt_df is not None:
        merged = pd.merge(submission, gt_df, on="PassengerId", suffixes=('_pred', '_true'))
        acc = accuracy_score(merged['Survived_true'], merged['Survived_pred'])
        print(f"Final Ensemble Accuracy: {acc:.5f}")

if __name__ == "__main__":
    main()
