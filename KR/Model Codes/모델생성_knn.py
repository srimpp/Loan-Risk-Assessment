import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from astropy.io.misc.yaml import name
import pickle
plt.rc('font', family='malgun gothic')

df = pd.read_csv('basedata_onehot2.csv')
#print(df.head(3))
#print(df.keys())

x = np.array(df.loc[:, df.columns != 'Risk_Flag'])
y = np.array(df['Risk_Flag'])
print(x[:2])
print(y[:2])

"""
# train/test 분리없이 돌리기
print('KNeighborsRegressor ---')
kmodel = KNeighborsRegressor(n_neighbors = 10).fit(x, y)
kpred = kmodel.predict(x)
print('KNeighborsRegressor pred :', kpred[0])
print('k_r2 :', r2_score(y, kpred))  # k_r2 : 0.40948057740337374
"""

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

print(len(x_train))  # 176400
print(len(x_test))   # 75600
print(len(y_train))  # 176400
print(len(y_test))   # 75600
"""
train_acc = []
test_acc = []

neighbors_set = range(1, 30, 2)

for n_neighbors in neighbors_set:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train))
    test_acc.append(clf.score(x_test, y_test))
    modelname='knnmodel'+str(n_neighbors)+'.dat'
    pickle.dump(clf, open(modelname, 'wb'))
    
print('train 평균 정확도 :', np.mean(train_acc))  # train 평균 정확도 : 0.9044992441421013
print('test 평균 정확도 :', np.mean(test_acc))    # test 평균 정확도 : 0.8904475308641976

# 시각화
plt.plot(neighbors_set, train_acc, label = 'train 분류 정확도')
plt.plot(neighbors_set, test_acc, label = 'test 분류 정확도')
plt.ylabel('정확도')
plt.ylabel('k의 갯수')
plt.legend()
plt.show()
"""

# 분류 예측

model = KNeighborsClassifier(n_neighbors=16)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('예측값 :', y_pred)
print('실제값 :', y_test)
# 예측값 : [0 0 0 ... 1 1 0]
# 실제값 : [0 0 0 ... 0 0 0]
# 총 갯수:75600, 오류수:8004

print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print()
print('분류 정확도 확인 1 :')
print('%.3f'%accuracy_score(y_test, y_pred))  # 0.894

print('분류 정확도 확인 2 :')
con_mat = pd.crosstab(y_test, y_pred)
print(con_mat)
# col_0      0     1
# row_0             
# 0      62476  3820
# 1       4184  5120
print((con_mat[0][0] + con_mat[1][1]) / len(y_test))  # 0.8941269841269841

print('분류 정확도 확인 3 :')
print('test :', model.score(x_test, y_test))     # 0.8941269841269841
print('train :', model.score(x_train, y_train))  # 0.9006689342403628

"""
# 모델 저장

pickle.dump(model, open('testmodel2.dat', 'wb'))

del model
read_model = pickle.load(open('testmodel2.dat', 'rb'))
"""
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
# [[62476  3820]
#  [ 4184  5120]]

from sklearn.metrics._plot.roc_curve import plot_roc_curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()

