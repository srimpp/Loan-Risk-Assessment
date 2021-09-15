from sklearn.preprocessing._data import StandardScaler
print('----- SVM -----')
import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from sklearn import svm
import pickle
from matplotlib import pyplot as plt
# SVM의 정의
# 데이터를 나눌 수 있는 구분선을 그어 패턴들과의 거리(마진)을 최대화하는 방법

# SVM 모델의 튜닝
# GridSearchCV : C와 gamma 값을 바꿔가면서 최상의 파라미터 찾기

# x, y 설정
data=pd.read_csv('basedata_onehot2.csv')
x=data[data.columns.drop('Risk_Flag')]
y = data['Risk_Flag']
# train/test로 분리 (7:3)
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size = 0.3, random_state = 123)
print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape) 
# (176400, 10) (75600, 10) (176400,) (75600,)
# (176400, 405) (75600, 405) (176400,) (75600,)
#data, 30996=1, 221004=0, y1_test: 9262 1s, y_train: 21734 1s, 154666 0s
#print(y.head(3))
#print(np.sum(y1_test==1))
#print(np.sum(y1_train==1))

#exit()
#model_svm1 = svm.SVC(C=0.1)
#model_svm1 = svm.SVC(C=100)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x1_train)
x_test_scaled = scaler.transform(x1_test)
#
# values = [.01, .1, 1, 10, 100, 1000]
# for v in values:
#     model=svm.LinearSVC(C=v, dual=False)
#     model.fit(x_train_scaled, y1_train)
#     pred = model.predict(x_test_scaled)
#     print('총 개수:%d, 오류수:%d'%(len(y1_test), (y1_test != pred).sum()))
#     print('분류 정확도 :', '%.3f'%accuracy_score(y1_test, pred))
#     print('C가',v,'일때 confusion:\n', confusion_matrix(y1_test, pred))
# exit()


model_svm1=svm.LinearSVC(C=.01, dual=False)
model_svm1.fit(x_train_scaled, y1_train) 

y_pred=model_svm1.predict(x_test_scaled)

print('총 갯수:%d, 오류수:%d'%(len(y1_test), (y1_test!=y_pred).sum()))
print('분류 정확도 확인1:')
print('%.3f'%accuracy_score(y1_test, y_pred))
print(confusion_matrix(y1_test, y_pred))
print('train',model_svm1.score(x_train_scaled, y1_train))
print('test',model_svm1.score(x_test_scaled, y1_test))
pickle.dump(model_svm1, open('model_svm_full_scal.dat', 'wb'))
exit()

"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
svc = svm.LinearSVC()
kfold = KFold(n_splits=5, shuffle=True, random_state=0) 
values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C':values}

# GridSearch
grid_search = GridSearchCV(svc, param_grid, cv=kfold)
grid_search.fit(x1_train, y1_train)

# result
print('최적 파라미터 점수 ==> {:.3f}'.format(grid_search.best_score_))
print('최적 파라미터 ==> {}'.format(grid_search.best_params_))
print('최적 파라미터 테스트의 점수 ==> {:.3f}'.format(grid_search.score(x1_test, y1_test)))

results = pd.DataFrame(grid_search.cv_results_)

print('results \n{}'.format(results.head()))





pickle.dump(model_svm1, open('svmmodelv3.dat', 'wb'))
"""
"""
#print('loading...')
model_svm1=pickle.load(open('model_svm_full.dat', 'rb'))
# 분류 예측
y_pred1 = model_svm1.predict(x1_test)
print('예측값 : ', y_pred1[:10])
print('실제값 : ', y1_test[:10])

print('총 개수:%d, 오류수:%d'%(len(y1_test), (y1_test != y_pred1).sum()))
print('분류 정확도 :', '%.3f'%accuracy_score(y1_test, y_pred1))
print('train',model_svm1.score(x1_train, y1_train))
print('test',model_svm1.score(x1_test, y1_test))
# 변수 10개로 모델 만들었을 때 => 총 개수:75600, 오류수:9262, 분류 정확도 : 0.877



y_pred=model_svm1.predict(x1_test)
print(confusion_matrix(y1_test, y_pred))

#ROC curve
testplot=plot_roc_curve(model_svm1, x1_test, y1_test)
plt.show()

#일반 SVM, C=100
#총 개수:75600, 오류수:9262
#분류 정확도 : 0.877
#[[66338     0]
# [ 9262     0]]
"""


#scaled linear svm
#[[66383     0]
# [ 9217     0]]
#train 0.8765362811791383
#test 0.8780820105820106
