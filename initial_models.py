import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Splits feature matrix and targets into training and test sets
def set_split(features, targets, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=test_ratio)
    return x_train, x_test, y_train, y_test


def save_model(model, model_type, target_identifier):
    pkl_filename = "models/" + model_type + "_" + target_identifier + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


# Decision tree model
def tree_model(features, targets, target_identifier):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(features, targets)
    save_model(clf, "decision_tree_model", target_identifier)


# Random forest model
def random_forest_model(features, targets, target_identifier):
    clf = RandomForestRegressor()
    clf.fit(features, targets)
    save_model(clf, "random_forest_model", target_identifier)


# GBM model
def gbm_model(features, targets, target_identifier):
    clf = GradientBoostingRegressor(random_state=0)
    clf.fit(features, targets)
    save_model(clf, "gbm_model", target_identifier)


updated_plays = pd.read_csv("updated_plays.csv")

# Some data cleaning
cleaned_plays = updated_plays
cleaned_plays.drop(['Unnamed: 0', 'gameId', 'playId', 'playDescription', 'personnelO', 'personnelD', 'penaltyCodes',
                    'penaltyJerseyNumbers', 'isDefensivePI', 'gameClock', 'possessionTeam', 'preSnapVisitorScore',
                    'yardlineSide', 'touchDown', 'preSnapHomeScore'], axis=1, inplace=True)

# Creating feature matrix
feature_matrix = cleaned_plays.drop(['playType', 'passResult', 'offensePlayResult',
                                     'playResult', 'epa', 'incompletePass', 'firstDown', 'yardlineNumber',
                                     'numDLoffense', 'numQBoffense', 'numDBoffense' ,'numTEdefense', 'numWRdefense', 'numOLdefense'],
                                    axis=1, inplace=False)

# Creating different target vectors
playResult_target = cleaned_plays['playResult']
epa_target = cleaned_plays['epa']


###################
# Linear Model work
###################
np.random.seed(0)
train_indices = np.random.rand(len(feature_matrix)) < 0.75
dummied_df = pd.get_dummies(feature_matrix)
x_train, x_test = dummied_df[train_indices], dummied_df[~train_indices]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''
linear_results = {}
for target in ['playResult', 'EPA']:
    if target == 'playResult':
        y_train, y_test = playResult_target[train_indices], playResult_target[~train_indices]
        alpha = 0.1 # Found from other tuning
    else:
        y_train, y_test = epa_target[train_indices], epa_target[~train_indices]
        alpha = 0.01 # Found from other tuning
    reg = LinearRegression().fit(x_train, y_train)
    lasso = Lasso(alpha=alpha).fit(x_train, y_train)
    linear_results[target + '_linReg_train'] = [round(mean_squared_error(y_train, reg.predict(x_train), squared=False),3), pd.DataFrame(reg.coef_, dummied_df.columns)]
    linear_results[target + '_linReg_test'] = [round(mean_squared_error(y_test, reg.predict(x_test), squared=False),3), pd.DataFrame(reg.coef_, dummied_df.columns)]
    linear_results[target + '_lasso_train'] = [round(mean_squared_error(y_train, lasso.predict(x_train), squared=False),3), pd.DataFrame(lasso.coef_, dummied_df.columns)]
    linear_results[target + '_lasso_test'] = [round(mean_squared_error(y_test, lasso.predict(x_test), squared=False),3), pd.DataFrame(lasso.coef_, dummied_df.columns)]
'''

###################
# Tree Model work & GBM
###################
y_train, y_test = epa_target[train_indices], epa_target[~train_indices]

tree_model(x_train, y_train, 'epa')
random_forest_model(x_train, y_train, 'epa')
gbm_model(x_train, y_train, 'epa')


