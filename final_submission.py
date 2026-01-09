import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
import os
from genetic_utils import get_genetic_feature

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

    # 2. Genetic Feature
    all_data['Genetic_Score'] = get_genetic_feature(all_data)

    # 3. Imputation
    all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
    all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
    all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

    # 3. Feature Engineering
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['IsAlone'] = 1
    all_data.loc[all_data['FamilySize'] > 1, 'IsAlone'] = 0

    # Title Extraction & Grouping (Restoring robust mapping)
    all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

    # 4. Binning (Using labels=False to ensure ordinal integers)
    all_data['FareBin_Code'] = pd.qcut(all_data['Fare'], 4, labels=False)
    all_data['AgeBin_Code'] = pd.cut(all_data['Age'].astype(int), 5, labels=False)

    # 5. Encoding
    label = LabelEncoder()
    all_data['Sex_Code'] = label.fit_transform(all_data['Sex'])
    all_data['Embarked_Code'] = label.fit_transform(all_data['Embarked'])
    all_data['Title_Code'] = label.fit_transform(all_data['Title'])

    # Features
    features = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code',
                'FamilySize', 'AgeBin_Code', 'FareBin_Code',
                'Family_Survival', 'Genetic_Score']

    X_train = all_data[:ntrain][features]
    X_test = all_data[ntrain:][features]
    y_train = train_df['Survived']

    return X_train, X_test, y_train

def main():
    train_df, test_df, gt_df = load_data()
    X_train, X_test, y_train = preprocess(train_df, test_df)

    # Models

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)

    # 2. XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=0, eval_metric='logloss')

    # 3. Gradient Boosting
    gbc = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=0)

    # 4. AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=0)

    # 5. Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=0)

    # Voting
    voting = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('gbc', gbc),
            ('ada', ada),
            ('dt', dt)
        ],
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
