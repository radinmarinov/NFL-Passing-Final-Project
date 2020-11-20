import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import tree


# Splits feature matrix and targets into training and test sets
def set_split(features, targets, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=test_ratio)
    return x_train, x_test, y_train, y_test


# Decision tree model
def tree_model(features, targets):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, targets)

    pkl_filename = "models/decision_tree_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)


updated_plays = pd.read_csv("updated_plays.csv")

# Some data cleaning
cleaned_plays = updated_plays
cleaned_plays.drop(['gameId', 'playId', 'playDescription', 'personnelO', 'personnelD', 'penaltyCodes',
                    'penaltyJerseyNumbers', 'isDefensivePI', 'gameClock', 'possessionTeam',
                    'yardlineSide', 'touchDown'], axis=1, inplace=True)

# Creating feature matrix
feature_matrix = cleaned_plays.drop(['playType', 'passResult', 'offensePlayResult',
                                     'playResult', 'epa', 'incompletePass', 'firstDown'],
                                    axis=1, inplace=False)

# Creating different target vectors
playResult_target = cleaned_plays['playResult']
epa_target = cleaned_plays['epa']
