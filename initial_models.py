import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

'''
# Splits feature matrix and targets into training and test sets
def set_split(features, targets, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=test_ratio)
    return x_train, x_test, y_train, y_test
'''


def save_model(model, model_type, target_identifier):
    pkl_filename = "models/" + model_type + "_" + target_identifier + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_type, target_identifier):
    pkl_filename = "models/" + model_type + "_" + target_identifier + ".pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        return model


# Decision tree model
def tree_model(features, targets, target_identifier):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(features, targets)
    save_model(clf, "decision_tree", target_identifier)


# Random forest model
def random_forest_model(features, targets, target_identifier):
    clf = RandomForestRegressor()
    clf.fit(features, targets)
    save_model(clf, "random_forest", target_identifier)


# GBM model
def gbm_model(features, targets, target_identifier):
    clf = GradientBoostingRegressor(random_state=0)
    clf.fit(features, targets)
    save_model(clf, "gbm", target_identifier)


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

##########
# Modeling
##########
np.random.seed(0)
train_indices = np.random.rand(len(feature_matrix)) < 0.75
dummied_df = pd.get_dummies(feature_matrix)
x_train, x_test = dummied_df[train_indices], dummied_df[~train_indices]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train, y_test = epa_target[train_indices], epa_target[~train_indices]

###################
# Linear Model work
###################
lin_reg_importance = []
lasso_importance = []

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

    if target == 'EPA':
        lin_reg_importance = reg.coef_
        lasso_importance = lasso.coef_

'''
plt.figure(3)
plt.bar([x for x in range(len(lin_reg_importance))], lin_reg_importance)
plt.xlabel('Feauture Number')
plt.ylabel('Importance')
plt.title('Feature Number v. Linear Regression Coefficients')
plt.savefig("graphs/lin_reg_importance")

plt.figure(4)
plt.bar([x for x in range(len(lasso_importance))], lasso_importance)
plt.xlabel('Feauture Number')
plt.ylabel('Importance')
plt.title('Feature Number v. Lasso Coefficients')
plt.savefig("graphs/lasso_importance")
'''

###################
# Tree Model work & GBM
###################

tree_model(x_train, y_train, 'epa')
random_forest_model(x_train, y_train, 'epa')
gbm_model(x_train, y_train, 'epa')


dt_clf = load_model("decision_tree", "epa")
rf_clf = load_model("random_forest", "epa")
gbm_clf = load_model("gbm", "epa")

dt_pred = dt_clf.predict(x_test)
rf_pred = rf_clf.predict(x_test)
gbm_pred = gbm_clf.predict(x_test)

tree_results_mse = {}
tree_results_acc = {}
tree_results_mse['epa_dt'] = (round(mean_squared_error(y_train, dt_clf.predict(x_train), squared=False), 3), round(mean_squared_error(y_test, dt_clf.predict(x_test), squared=False), 3))
tree_results_mse['epa_rf'] = (round(mean_squared_error(y_train, rf_clf.predict(x_train), squared=False), 3), round(mean_squared_error(y_test, rf_clf.predict(x_test), squared=False), 3))
tree_results_mse['epa_gbm'] = (round(mean_squared_error(y_train, gbm_clf.predict(x_train), squared=False), 3), round(mean_squared_error(y_test, gbm_clf.predict(x_test), squared=False), 3))

tree_results_acc['epa_dt'] = dt_clf.score(x_test, y_test)
tree_results_acc['epa_rf'] = rf_clf.score(x_test, y_test)
tree_results_acc['epa_gbm'] = gbm_clf.score(x_test, y_test)

dt_importance = dt_clf.feature_importances_
'''
plt.figure(1)
plt.bar([x for x in range(len(dt_importance))], dt_importance)
plt.xlabel('Feauture Number')
plt.ylabel('Importance')
plt.title('Feature Number v. Decision Tree Importance')
plt.savefig("graphs/dt_importance")
'''

rf_importance = rf_clf.feature_importances_
'''
plt.figure(2)
plt.bar([x for x in range(len(rf_importance))], rf_importance)
plt.xlabel('Feauture Number')
plt.ylabel('Importance')
plt.title('Feature Number v. Random Forest Importance')
plt.savefig("graphs/rf_importance")
'''

gbm_importance = gbm_clf.feature_importances_
'''
plt.figure(5)
plt.bar([x for x in range(len(gbm_importance))], gbm_importance)
plt.xlabel('Feauture Number')
plt.ylabel('Importance')
plt.title('Feature Number v. GBM Importance')
plt.savefig("graphs/gbm_importance")
'''

'''
for i, col in enumerate(feature_matrix.columns):
    print(i, col)
'''

################
# Neural Network 
################
'''
hidden_layer = (2,2,2)
activation = 'tanh'
MLP = MLPRegressor(hidden_layer_sizes=hidden_layer
                    , random_state=1
                    , activation=activation
                    , max_iter=200).fit(x_train, y_train)
print(round(mean_squared_error(y_train, MLP.predict(x_train), squared=False),3), round(mean_squared_error(y_test, MLP.predict(x_test), squared=False),3))
'''

