from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import pickle

pd.set_option('max_columns', None)
data = pd.read_csv("../testdata/basedata_onehot2.csv")
#data = pd.read_csv("../testdata/simpledata_career.csv")

# import re
# data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

x = data.drop(['Risk_Flag'], axis=1) #독립변수
y = data['Risk_Flag'] #종속변수

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 123) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(176400, 11) (75600, 11) (176400,) (75600,)

#표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print('--------------------------------모델---------------------------------------------------------------------------------------')
#LogisticRegression
# model = LogisticRegression()
# model.fit(x_train_scaled, y_train)
#
# pred = model.predict(x_test_scaled)
# print('LogisticRegression: %.2f'%(metrics.accuracy_score(y_test, pred)*100),'%')
'''
#LogisticRegression GridSearchCV 이용 최적 파라미터값 선택
parameters = {'C':[0.01,0.1,1,10,100]}

grid_search = GridSearchCV(model, param_grid = parameters, cv=5, refit=True) #5번학습, 최적의 파라미터가 나올떄까지 반복학습한다
grid_search.fit(x_train_scaled, y_train)
print('GridSearchCV의 최적 파라미터 : ', grid_search.best_params_) #{'C': 0.01}
print('GridSearchCV의 최고 정확도 : ', grid_search.best_score_) #0.8765362811791384
'''
#KNeighborsClassifier모델 생성
# model = KNeighborsClassifier()
# model.fit(x_train_scaled, y_train)
#
# pred = model.predict(x_test_scaled)
# print('KNeighborsClassifier: %.2f'%(metrics.accuracy_score(y_test, pred)*100),'%')
'''
#KNeighborsClassifier GridSearchCV 이용 최적 파라미터값 선택
parameters = {'n_neighbors':[1,2,3,4,5]}

grid_search = GridSearchCV(model, param_grid = parameters, cv=5, refit=True) #5번학습, 최적의 파라미터가 나올떄까지 반복학습한다
grid_search.fit(x_train_scaled, y_train)
print('GridSearchCV의 최적 파라미터 : ', grid_search.best_params_) #{'n_neighbors': 5}
print('GridSearchCV의 최고 정확도 : ', grid_search.best_score_) #0.8890986394557825
'''
# XGBClassifier
# model = XGBClassifier(random_state=1)
# model.fit(x_train_scaled, y_train)
#
# pred = model.predict(x_test_scaled)
# print('XGBClassifier: %.2f'%(metrics.accuracy_score(y_test, pred)*100),'%')

'''
#XGBClassifier GridSearchCV 이용 최적 파라미터값 선택
parameters = {'max_depth':[3,5,10],'n_estimators':[5,10,100,200,300]}

grid_search = GridSearchCV(model, param_grid = parameters, cv=5, refit=True) #5번학습, 최적의 파라미터가 나올떄까지 반복학습한다
grid_search.fit(x_train_scaled, y_train)
print('GridSearchCV의 최적 파라미터 : ', grid_search.best_params_) #{'max_depth': 10, 'n_estimators': 300}
print('GridSearchCV의 최고 정확도 : ', grid_search.best_score_) #0.8962755102040816
'''
# LGBMClassifier
# model = LGBMClassifier(random_state=1)
# model.fit(x_train_scaled, y_train)
#
# pred = model.predict(x_test_scaled)
# print('LGBMClassifier: %.2f'%(metrics.accuracy_score(y_test, pred)*100),'%')

'''
#LGBMClassifier GridSearchCV 이용 최적 파라미터값 선택
parameters = {'max_depth':[10,30,50],'n_estimators':[100,500,3000,10000]}

grid_search = GridSearchCV(model, param_grid = parameters, cv=5, refit=True) #5번학습, 최적의 파라미터가 나올떄까지 반복학습한다
grid_search.fit(x_train_scaled, y_train)
print('GridSearchCV의 최적 파라미터 : ', grid_search.best_params_) #{'max_depth': 30, 'n_estimators': 3000}
print('GridSearchCV의 최고 정확도 : ', grid_search.best_score_) #0.8965759637188209
'''
'''
#Cross Validation : test all model
models = {
#    'LogisticRegression' : LogisticRegression(C=0.01),
#    'KNeighborsClassifier' : KNeighborsClassifier(n_neighbors=5),
#    'XGBClassifier' : XGBClassifier(max_depth = 10, n_estimators = 300, random_state=1),
    'LGBMClassifier' : LGBMClassifier(max_depth = 10, n_estimators = 3000, random_state=1)
}
 
cv = KFold(n_splits=5, random_state=1)
 
for name, model in models.items():
    #scores = cross_val_score(model, x, y, cv=cv)
    scores=cross_val_score(model, x.values, y.values, cv=cv) #xgb는 feature name이 바뀌므로 이렇게
 
    print('Cross validation_%s: %.2f'%(name, np.mean(scores) * 100),'%')
'''

print('--------------------------모델 선택-----------------------------------------------------------------------------------------')
'''
#로지스틱회귀에서 주요변수 뽑아서 만듬
corresult=x.corrwith(y)
print(corresult.head(3), type(corresult))
print(max(corresult.abs()))
abscor=corresult.abs()
print(abscor.sort_values(ascending=False).head(15))
print('-----------')
print(corresult.sort_values(ascending=False).head(15))
print('------------')
print(corresult.sort_values(ascending=True).head(15))

a = data[['Experience', 'Bhubaneswar', 'rented', 'Kochi','Car_Ownership','owned','Madhya_Pradesh','Gwalior',
               'Age','Married/Single','Buxar[37]','Barasat','Kerala','Satna','Sikar']] #독립변수
b = data['Risk_Flag'] #종속변수

x_train, x_test, y_train, y_test = train_test_split(a,b, test_size = 0.3, random_state = 123) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(176400, 11) (75600, 11) (176400,) (75600,)

#표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
'''

#선택모델생성
#model = GaussianNB()
#model = LogisticRegression(C=0.01)
#model = XGBClassifier(max_depth = 10, n_estimators = 300, random_state=1)
model = LGBMClassifier(max_depth = 10, n_estimators = 1000, random_state=1)
#model = RandomForestClassifier(n_estimators = 500)

model.fit(x_train, y_train)
pred = model.predict(x_test)

# model 정확도
print('선택모델 정확도: %.2f'%(metrics.accuracy_score(y_test, pred)*100),'%')
print('test_accuracy : %.2f'%(accuracy_score(y_test, pred)*100),'%') 
print('train_accuracy : %.2f'%(accuracy_score(y_train, model.predict(x_train))*100),'%')

print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != pred).sum()))
#print('roc_auc_score: %.2f'%(roc_auc_score(y_test, pred)*100),'%')

#confusion_matrix
print('confusion_matrix: \n',confusion_matrix(y_test, pred))

#ROC
from sklearn.metrics._plot.roc_curve import plot_roc_curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()


# from lightgbm import plot_importance
# fig, ax = plt.subplots(1,1, figsize=(10,8))
# plot_importance(model, ax=ax, height=0.4, max_num_features=20)
# plt.show()

print('---------------------------------------------------------------------------------------')
# train_test 정확도비교
# train_acc = []
# test_acc = []
#
# estimators_set = range(100, 5000, 100)
#
# for n_estimators in estimators_set:
#     clf = LGBMClassifier(max_depth = 30, n_estimators = n_estimators, random_state=1)
#     clf.fit(x_train, y_train)
#     train_acc.append(clf.score(x_train, y_train))
#     test_acc.append(clf.score(x_test, y_test))
#
# print('train 평균 정확도 :', np.mean(train_acc))  # train 평균 정확도 : 0.9044992441421013
# print('test 평균 정확도 :', np.mean(test_acc))    # test 평균 정확도 : 0.8904475308641976
#
# # 시각화
# plt.plot(estimators_set, train_acc, label = 'train_acc')
# plt.plot(estimators_set, test_acc, label = 'test_acc')
# plt.xlabel('n_estimators')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()

'''
# 학습된 모델 저장
fileName = 'LGBMClassifier_model.sav'
pickle.dump(model, open(fileName, 'wb'))

# 학급된 모델 읽기
model = pickle.load(open('LGBMClassifier_model.sav', 'rb'))
'''

# 결정계수확인
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, pred)
# print(r2)


