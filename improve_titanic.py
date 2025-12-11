import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

def preprocess(df):
    df = df.copy()

    # Fill missing Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fill missing Fare
    df['Fare'] = df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

    # Extract Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Group titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Map Title
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    # Convert Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Fill missing Age
    # Using Pclass, Sex, Title to impute Age
    df['Age'] = df.groupby(['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Cabin - Extract Deck
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    # Grouping
    df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
    df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')

    # Encode Deck
    le_deck = LabelEncoder()
    df['Deck'] = le_deck.fit_transform(df['Deck'])

    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Embarked mapping
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Create Age*Class
    df['Age*Class'] = df['Age'] * df['Pclass']

    # Drop unused
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return df

def main():
    train_df, test_df, gt_df = load_data()

    ntrain = train_df.shape[0]

    y_train = train_df['Survived']
    all_data = pd.concat([train_df.drop(['Survived'], axis=1), test_df]).reset_index(drop=True)

    all_data = preprocess(all_data)

    X_train = all_data[:ntrain]
    X_test = all_data[ntrain:]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    # Save submission
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'")

    # Evaluate if ground truth is available
    if gt_df is not None:
        merged = pd.merge(submission, gt_df, on="PassengerId", suffixes=('_pred', '_true'))
        acc = accuracy_score(merged['Survived_true'], merged['Survived_pred'])
        print(f"Final Accuracy: {acc:.5f}")
    else:
        print("Ground truth file not found. Evaluation skipped.")

if __name__ == "__main__":
    main()
