#from loan_data import loan_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import pickle
import numpy as np
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from matplotlib.pyplot import tight_layout

pd.set_option('max_columns', None)

#----------import data-------------------
data=pd.read_csv('simpledata_career.csv')
#data=pd.read_csv('basedata_onehot2.csv')
#print(data.shape, data.head(3))


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
#simple ver: test_acc=.90011, train_acc=.93708
#모델 저장 ---------------------------

pickle.dump(model, open('rfmodel_simple.dat', 'wb'))

"""
print(x_test.shape)
#del model
model=pickle.load(open('rfmodel_simple.dat', 'rb'))

#confusion matrix
y_pred=model.predict(x_test.values)
#print(np.where(y_pred==1)) #1로 분류된 케이스 인덱스 확인
#csv로 저장
#x_test.to_csv('x_test_crossval.csv',sep=',', index=False, encoding='utf-8-sig')
#4, 13, 21, 75572, 75575, 75590 가 1로 분류되었다, 엑셀에서 6번째 income:270761
print(confusion_matrix(y_test, y_pred))
print('test', model.score(x_test, y_test))
print('train',model.score(x_train, y_train)) #두 개의 값 차이가 크면 과적합 의심





#중요도 그래프
import matplotlib.pyplot as plt
"""
#print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
#print('ordered', np.sort(model.feature_importances_))
plt.rc('font',family='Malgun Gothic')

plt.figure(figsize=(8,6))
importances=pd.Series(model.feature_importances_, index=x.columns).nlargest(10)
labels=importances.index
# import textwrap
# f=lambda x:textwrap.fill(x.get_text(), 10)
# labels=map(f, labels)
#print(labels)
max_chars=23
labels=['\n'.join(label[i:i + max_chars ] 
                        for i in range(0, len(label), max_chars ))
              for label in labels]

#importances.nlargest(10).plot(kind='barh')
plt.barh(y=labels, width=importances)

plt.xlabel('Variable Importance')
plt.title('변수 중요도')
#plt.xlim(0, .3)
plt.show()



#중요도 제일 높은 10개 인덱스 추출
# topind=np.argpartition(model.feature_importances_, -10)[-10:]
# print(model.feature_importances_[topind])
# print(x.columns[topind])
#print(x[topind])

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



#ROC curve
testplot=plot_roc_curve(model, x_test, y_test)
plt.show()


#[[63067  3271]
# [ 4281  4981]]

"""