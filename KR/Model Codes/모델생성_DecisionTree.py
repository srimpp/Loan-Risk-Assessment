print('----- Decision Tree -----')
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Decision tree의 정의
# 지도 학습 모델 중 하나이며, 일련의 분류 규칙을 통해 데이터를 분류, 또는 회귀하는 모델이다.
# 모델이 Tree 구조를 가지고 있다.
# 트리에서 각 질문을 노드(Node)라고 한다. 맨 처음 노드를 Root Node, 중간 분류 기준을 Intermediate Node, 맨 마지막 노드를 Terminal Node라고 한다.
# 불순도(impurity)를 낮추는 것을 목표로 하며, 불순도는 한 범주 안에 다른 데이터가 얼마나 섞여 있는지를 의미한다. (아래 모델에서는 엔트로피 지수(Entropy)를 기준으로 사용)


# Decision tree 모델의 튜닝
# 1) 특성 중요도를 기준으로 중요 변수를 추출하여 새로운 모델을 만들어, 정확도를 비교한다.
# 2) GridSearchCV :  max_depth, min_samples_split, splitter 등을 조정한다. (최적의 파라미터 찾기)

print('----- 산업군 미포함 data를 이용 -----')
import pandas as pd
data1 = pd.read_csv("basedata_onehot2.csv")


# x, y 설정
x1 = data1.drop(['Risk_Flag'], axis = 1) # Columns: 405 entries
#print(x1.info())
y1 = data1['Risk_Flag']
#print(y1.unique())

# train/test로 분리 (7:3)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.3, random_state = 0)
print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape) # (176400, 407) (75600, 407) (176400,) (75600,)

# 모델 
model1 = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', max_depth = 4, min_samples_split = 2, random_state = 0)  
model1.fit(x1_train, y1_train) # 모델 학습 진행

#GridSearchCV의 최적 파라미터 :  {'max_depth': 4, 'min_samples_split': 2, 'splitter': 'best'(default 값)}

# 분류 예측
y_pred1 = model1.predict(x1_test)

print('예측값 : ', y_pred1[:10])
print('실제값 : ', y1_test[:10])

y_pred11 = model1.predict(x1_train)
print('총 개수:%d, 오류수:%d'%(len(y1_test), (y1_test != y_pred1).sum()))
print('분류 정확도(train) :', '%.3f'%accuracy_score(y1_train, y_pred11))
print('분류 정확도(test) :', '%.3f'%accuracy_score(y1_test, y_pred1))

# 총 개수:75600, 오류수:9236, 분류 정확도 : 0.878
# 분류 정확도(train) : 0.877
# 분류 정확도(test) : 0.878


# Decision Tree를 시각화
from sklearn.tree import export_graphviz 
import graphviz

print('--- 모델 튜닝 1) 특성 중요도 출력---')
#print(model1.feature_importances_)
df1 = pd.DataFrame({'특성' : x1.columns ,'특성중요도': model1.feature_importances_})
df1_sort_top10 = df1.sort_values(by="특성중요도", ascending=False).head(12)
print(df1_sort_top10)

from matplotlib import pyplot as plt
plt.rc('font', family='malgun gothic')
plt.barh(df1_sort_top10['특성'], df1_sort_top10['특성중요도']) 
plt.xlabel("특성중요도") 
plt.ylabel("특성") 
plt.title("Decision Tree 모델의 특성 중요도")
plt.show()


# 중요한 특성 12개로 새로운 모델 생성
print('----- 특성 12개로 모델 생성 (산업군 미포함 data를 이용)----')

# x, y 설정
x2 = data1[['Experience', 'Ratlam', 'Kerala', 'rented', 'Purnia[26]', 'Bhubaneswar', 'Bhavnagar', 'Kozhikode', 'Jamnagar', 'Surgeon', 'Computer_hardware_engineer', 'Flight_attendant']]
y2 = data1['Risk_Flag']

# train/test로 분리 (7:3)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = 0.3, random_state = 0)
print(x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape) # (176400, 7) (75600, 7) (176400,) (75600,)

# 모델 
model2 = DecisionTreeClassifier(criterion = 'entropy',  splitter = 'best' ,max_depth = 4, min_samples_split = 2, random_state = 0)  
model2.fit(x2_train, y2_train) # 모델 학습 진행

# 분류 예측
y_pred2 = model2.predict(x2_test)
y_pred22 = model2.predict(x2_train)
print('예측값 : ', y_pred2[:10])
print('실제값 : ', y2_test[:10])

print('총 개수:%d, 오류수:%d'%(len(y2_test), (y2_test != y_pred2).sum()))
print('분류 정확도(train) :', '%.3f'%accuracy_score(y2_train, y_pred22))
print('분류 정확도(test) :', '%.3f'%accuracy_score(y2_test, y_pred2))


# 변수 12개 이용 =>  개수:75600, 오류수:9223
# 분류 정확도(train) : 0.877
# 분류 정확도(test) : 0.878
# 모든 변수를 사용했을 때와, 12개의 변수를 사용했을 때 정확도는 차이가 없다. 
# 특성 12개만을 이용해서 대출 risk 여부를 예측할 수 있다. 


print('----- 모델 튜닝 2)  GridSearchCV -----')

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth' : [1,2,3,4], 'min_samples_split' : [2,3,4], 'splitter': ['best', 'random']}

grid_dtree = GridSearchCV(model2, param_grid = parameters, cv=5, refit = True) # 최적의 파라미터가 나올 때까지 반복 학습
grid_dtree.fit(x2_train, y2_train)

pd.set_option('max_columns', None)
score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df)

print('GridSearchCV의 최적 파라미터 : ', grid_dtree.best_params_)
print('GridSearchCV의 최고 정확도 : ', grid_dtree.best_score_)

# GridSearchCV로 찾은 최적 파라미터 :  {'max_depth': 4, 'min_samples_split': 2, 'splitter': 'best'}


# 파일 생성(전체 변수)
#export_graphviz(model1, out_file="loan1.png",  \
#                feature_names = x1_train.columns, class_names=["0", "1"], impurity=True, filled=True) 

# 파일 생성(12개 변수 사용)
export_graphviz(model2, out_file="loan2.png",  \
                feature_names =x2_train.columns, class_names=["0", "1"], impurity=True, filled=True) 
with open("loan2.png") as f:
    dot_graph = f.read()
graph = graphviz.Source(dot_graph, format='png')
graph.view()

# K-Fold 교차 검증
from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(model2, x2_train, y2_train, scoring='accuracy', cv=5)
print('교차 검증별 정확도 : ', np.round(scores,4))
print('평균 검증 정확도 : ', np.round(np.mean(scores), 4))

# 교차 검증별 정확도 :  [0.8774 0.8773 0.8772 0.8771 0.8771]
# 평균 검증 정확도 :  0.8772


# confusion-matrix
from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y1_test, y_pred1)) # 모든 변수 사용
print(confusion_matrix(y2_test, y_pred2))   # 12개 변수 사용

#ROC
from matplotlib import pyplot as plt 
from sklearn.metrics._plot.roc_curve import plot_roc_curve
# testplot=plot_roc_curve(model2, x2_test, y2_test) # 모든 변수 사용
testplot=plot_roc_curve(model2, x2_test, y2_test) # 12개 변수만 사용
plt.show()

"""
# 모델 저장 
import pickle
pickle.dump(model2, open('tree_model.sav', 'wb'))
read_model = pickle.load(open('tree_model.sav', 'rb'))
"""


