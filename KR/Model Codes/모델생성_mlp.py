import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
import mglearn
import pickle
from statsmodels.sandbox.tools import cross_val


# 데이터 가져오기 
datas = pd.read_csv('basedata_onehot2.csv')
pd.set_option('max_columns', None)
# print(datas.head())
# print(datas.info())
print(datas.shape) # (252000, 406)
print(datas.isnull().any()) 

# 데이터 분리
x_data = datas.drop(['Risk_Flag'], axis=1)
y_data = datas['Risk_Flag']
print(x_data.shape) 
print(y_data.shape) 

# train, test 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.3,
                                                    random_state=126,
                                                    shuffle=True)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (176400, 405) (75600, 405) (176400,) (75600,)

# 정규화
scaler = MinMaxScaler()
scaler.fit(x_train)

# Apply the scaler to the X training data
x_train = scaler.transform(x_train)
# Apply the scaler to the X test data
x_test = scaler.transform(x_test)

# k-fold validation
# kfold = KFold(n_splits = 5)


# Create a perceptron object with the parameters: 
# 1)100 iterations (epochs) over the data, and a learning rate of 0.1
# Create a perceptron object with the parameters: 
# 1)100 iterations (epochs) over the data, and a learning rate of 0.1
# model_mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10), 
#                           activation = 'relu',
#                           solver='sgd', 
#                           learning_rate_init=0.1, 
#                           random_state=126,
#                           early_stopping = True,
#                           validation_fraction = 0.3,
#                           n_iter_no_change = 20,
#                           verbose=2)

# model_mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), 
#                           solver='adam', 
#                           learning_rate_init=0.1, 
#                           verbose=2)
# Train the perceptron
# model_mlp.fit(x_train, y_train)



# Save the model
# pickle.dump(model_mlp, open('model_mlp.dat', 'wb'))


#del model
model_mlp=pickle.load(open('model_mlp.dat', 'rb'))



# Apply GridSearchCV
# parameter = {
#              'max_iter' : list(range(10,50,10)),
#              'hidden_layer_sizes': np.arange(1,5,1),
#              'learning_rate_init': [0.001, 0.01, 0.1],
#              }
# grid_mlp = GridSearchCV(MLPClassifier(), 
#                         param_grid=parameter,
#                         scoring='accuracy',
#                         cv=10,
#                         n_jobs=-1,
#                         refit=True)
#
# grid_mlp.fit(x_train, y_train)


# Find the best parameter
# print(grid_mlp.best_params_)
# print(grid_mlp.best_score_)
#{'hidden_layer_sizes': 4, 'learning_rate_init': 0.1, 'max_iter': 40, 'solver': 'sgd'}
# 0.8770804988662132


# Predict the model 
y_pred = model_mlp.predict(x_test)

# model 정확도
print('test_accuracy : ', accuracy_score(y_test, y_pred)) # accuracy :  0.8015740740740741
print('train_accuracy : ', accuracy_score(y_train, model_mlp.predict(x_train)))

# confusion matrix
print('confusion_matrix\n', confusion_matrix(y_test, y_pred))

# classification_report
print('classification_report\n', classification_report(y_test, y_pred))

# roc curve
mlp_roc_score = roc_auc_score(y_test, model_mlp.predict_proba(x_test)[:,1])
print('ROC AUC : {0:.4f}'.format(mlp_roc_score))


# 결과 값 
print('loss : ', model_mlp.loss_) 
print('score :', model_mlp.score(x_test, y_test))
print('prob : ', model_mlp.predict_proba(x_test))

scores = cross_val_score(model_mlp, x_test, y_test, cv=5)
print('각각의 정답률 : ', scores)
print('평균 정답률 : ', scores.mean())



# 시각화 
# roc 커브
mlp_disp = plot_roc_curve(model_mlp, x_test, y_test)
plt.show()

# confusion matrix 
mlp_conf = plot_confusion_matrix(model_mlp, x_test, y_test)
plt.show()














