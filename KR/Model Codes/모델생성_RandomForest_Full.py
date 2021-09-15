#from loan_data import loan_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import pickle
import numpy as np
from sklearn.metrics._plot.roc_curve import plot_roc_curve

pd.set_option('max_columns', None)

#----------import data--------
data=pd.read_csv('simpledata_career.csv')
#print(data.head(3))
#data=pd.read_csv('test_simpledata_career.csv')
print(data.shape)



#----------declare x, y, y:Risk Flag, 0:No History of default, 1:History of Default-----
x=data[data.columns.drop('Risk_Flag')]
#print(x.shape)
y=data['Risk_Flag']
#print(y.shape)

#-----------validation split, train:test=7:3----------
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.3,random_state=0)
"""
#------------Random Forest---------------------
model=RandomForestClassifier(criterion='entropy',n_estimators=500, n_jobs=-1, random_state=1)
#model=RandomForestClassifier(criterion='gini',n_estimators=500, n_jobs=2, random_state=1)
model.fit(x_train, y_train) #학습 진행

y_pred=model.predict(x_test)

print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test!=y_pred).sum()))
print('분류 정확도 확인1:')
print('%.3f'%accuracy_score(y_test, y_pred))


print('분류 정확도 확인 3:')
print('test', model.score(x_test, y_test))
print('train',model.score(x_train, y_train)) #두 개의 값 차이가 크면 과적합 의심

#entropy ver: test_acc=.90066, train_acc=.93708

#모델 저장 ---------------------------

pickle.dump(model, open('testmodel.dat', 'wb'))
"""

#del model
model=pickle.load(open('rfmodel_simple.dat', 'rb'))

"""
#-------------------------중요도 그래프-----------------------
import matplotlib.pyplot as plt
#print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
#print('ordered', np.sort(model.feature_importances_))

#중요도 제일 높은 10개 인덱스 추출
topind=np.argpartition(model.feature_importances_, -10)[-10:]
print(model.feature_importances_[topind])
print(x.columns[topind])
#print(x[topind])

importances=pd.Series(model.feature_importances_, index=x.columns)
importances.nlargest(10).plot(kind='barh')
plt.show()

# def plot_feature_importances(model):   # 특성 중요도 시각화
#     n_features = 10
#     plt.barh(range(n_features), model.feature_importances_[topind], align='center')
#     plt.yticks(np.arange(n_features), x.columns[topind])
#     plt.xlabel("attr importances")
#     plt.ylabel("attr")
#     plt.ylim(-1, n_features)
#     plt.show()
#     plt.close()
#
# plot_feature_importances(model)

#['Maharashtra', 'Andhra_Pradesh', 'Uttar_Pradesh', 'Married/Single',
#       'CURRENT_HOUSE_YRS', 'Car_Ownership', 'CURRENT_JOB_YRS', 'Age',
#       'Experience', 'Income']
"""



#confusion matrix
y_pred=model.predict(data)
print(y_pred)
print((y_pred==1).sum())
print((y_pred==0).sum())

print(np.where(y_pred==1))

print(data.iloc[np.where(y_pred==1)])
onedata=data.iloc[np.where(y_pred==1)]
print(onedata.shape)
"""
print(onedata['Age'].describe().astype('int'))
print(data1['Age'].describe().astype('int'))

print(onedata['Income'].describe().astype('int'))
print(data1['Income'].describe().astype('int'))

print(onedata['Experience'].describe().astype('int'))
print(data1['Experience'].describe().astype('int'))

print(onedata['CURRENT_HOUSE_YRS'].describe().astype('int'))
print(data1['CURRENT_HOUSE_YRS'].describe().astype('int'))

print(onedata['CURRENT_JOB_YRS'].describe().astype('int'))
print(data1['CURRENT_JOB_YRS'].describe().astype('int'))
"""

#onedata.to_csv('testdata_ones.csv',sep=',', index=False, encoding='utf-8-sig')
#test predicted 3130 1s



"""
print(confusion_matrix(y_test, y_pred))

#ROC curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()
"""
