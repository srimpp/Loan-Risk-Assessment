# Naive_Bayes

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
#             '../final/main/static/data_onehot_one.csv')
datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            '../final/main/static/basedata_onehot2.csv')

# datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
#              '../final/main/static/simpledata_career.csv')

data = pd.read_csv(datas)


x = data[data.columns.drop('Risk_Flag')]
y = data['Risk_Flag']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (189000, 405) (63000, 405) (189000,) (63000,)



#표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = GaussianNB()
#model = LogisticRegression(C=0.01)
#model = XGBClassifier(max_depth = 10, n_estimators = 300, random_state=1)
#model = LGBMClassifier(max_depth = 10, n_estimators = 1000, random_state=1)


model.fit(x_train, y_train)
pred = model.predict(x_test)

# model 정확도
print('선택모델 정확도: %.2f'%(metrics.accuracy_score(y_test, pred) * 100),'%')
print('test_accuracy : %.2f'%(accuracy_score(y_test, pred) * 100),'%') 
print('train_accuracy : %.2f'%(accuracy_score(y_train, model.predict(x_train)) * 100),'%')

print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != pred).sum()))
#print('roc_auc_score: %.2f'%(roc_auc_score(y_test, pred)100),'%')

#confusion_matrix
print('confusion_matrix: \n',confusion_matrix(y_test, pred))

#ROC
from sklearn.metrics._plot.roc_curve import plot_roc_curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()





















"""
# print(data)
# print(data.shape) # (252000, 406)
# print(data.isna().any()) # 0, 전처리에서 확인했지만 절차상 시행
# print(data.info())

x = data[data.columns.drop('Risk_Flag')]
y = data['Risk_Flag']

# print(x.shape) # (252000, 405)
# print(y.shape) # (252000,)



# train / test로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1, shuffle = True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (189000, 405) (63000, 405) (189000,) (63000,)

# 모델 불러오기
# model=pickle.load(open('Naive.dat', 'rb'))


# 방법1 : onehot 데이터
model = GaussianNB()
model.fit(x_train, y_train)

# 모델 저장
pickle.dump(model, open('Naive.dat', "wb"))



# 예측
pred = model.predict(x_test)
print(pred)

# 정확도
acc = accuracy_score(y_test, pred)
print('분류 정확도 : ', acc) # 분류 정확도 :  0.8783968253968254

# 일치하지 않는 개수 출력
print('일치하지 않는 데이터의 수 : ', (y_test != pred).sum())

# k-fold 교차 검증 방법으로 검증.
from sklearn import model_selection
cross_val = model_selection.cross_val_score(model, x_train, y_train, cv = 5)
print(cross_val) # [0.87700397 0.87700397 0.87700397 0.87700397 0.87698413], 5번 진행하였음에도 크게 달라진 점은 없다.



print(confusion_matrix(y_test, pred))

from sklearn.metrics._plot.roc_curve import plot_roc_curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()
"""
