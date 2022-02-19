import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

raw_data = pd.read_csv('Bank-data.csv')

data = raw_data.copy()
data = data.drop(['Unnamed: 0'], axis=1)
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data.describe()

# ## Model: Simple Classification
y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
results_log.summary()

plt.scatter(x1, y, color='C0')
plt.xlabel('Duration', fontsize=20)
plt.ylabel('Subscription', fontsize=20)
plt.show()

# ## Expanded model: Multivariate Classification
estimators = ['interest_rate', 'credit', 'march', 'previous', 'duration']
X1_all = data[estimators]
y = data['y']

X_all = sm.add_constant(X1_all)
reg_logit = sm.Logit(y, X_all)
results_logit = reg_logit.fit()
results_logit.summary2()


# ### Confusion Matrix


def confusion_matrix(data_array, actual_values, model):
    pred_values = model.predict(data_array)
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    return cm, accuracy


confusion_matrix(X_all, y, results_logit)

# ## Testing the accuracy of the model
raw_data2 = pd.read_csv('Bank-data-testing.csv')
data_test = raw_data2.copy()
data_test = data_test.drop(['Unnamed: 0'], axis=1)

data_test['y'] = data_test['y'].map({'yes': 1, 'no': 0})

y_test = data_test['y']
X1_test = data_test[estimators]
X_test = sm.add_constant(X1_test)

confusion_matrix(X_test, y_test, results_logit)
confusion_matrix(X_all, y, results_logit)
