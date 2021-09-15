# Tensor_Flow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 첫 번째 GPU에 4GB이상 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4300)]) # 4300
    except RuntimeError as e:
        # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)


# datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
#              '../final/main/static/data_onehot_one.csv')


datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            '../final/main/static/simpledata_career.csv')
data = pd.read_csv(datas)


# x = data.iloc[:, 0:62]
# x = x[x.columns.drop('Risk_Flag')]

x = data[data.columns.drop('Risk_Flag')]

y = data['Risk_Flag']


# train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (189000, 405) (63000, 405) (189000,) (63000,)
# print(x_test[:20])
# print(x_train[:20])

# (176400, 420) (75600, 420) (176400,) (75600,)


# 모델 불러오기
from tensorflow.keras.models import load_model
# model = load_model('tensor_logit.hdf5')



# 모델1
# """ 25
model = Sequential()
model.add(Dense(512, input_dim = 25, activation = 'relu')) # 61 or 407 or 420
model.add(Dense(256, activation = 'relu')) # relu도 사용 가능
model.add(Dropout(0.3))
model.add(BatchNormalization()) # 정규화

model.add(Dense(256, activation = 'relu')) # relu도 사용 가능
# model.add(Dropout(0.3))
model.add(BatchNormalization()) # 정규화

model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(1, activation = 'sigmoid'))

# EarlyStoping
es = EarlyStopping(monitor='val_loss', patience = 10) # 10 정도로 하라고 선생님이 말씀하심


# 체크 포인트
chk = ModelCheckpoint(filepath = 'tensor_simple.hdf5', monitor = 'val_loss', save_best_only=True)


# SGD, RMSprop, Adam, Adagrad
opti = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer = opti, loss = 'binary_crossentropy', metrics = ['acc']) # 'mae',  mae : 실제값과 예측값의 Error를 절대값으로 평균화


model.fit(x_train, y_train, epochs = 800, batch_size = 128, verbose = 2, validation_split = 0.3, callbacks = [es, chk])
# """
# history = model

print('모델 평가 : ', model.evaluate(x_test, y_test))
# print(loss, mse, acc)


# 예측값
pred = (model.predict(x_test))
print(pred.flatten()[:20])
pred = (pred.flatten() > 0.5).astype('int')
print('pred : ', pred[:20])

print('real : ', np.array(y_train)[:20])
print('일치 하지 않는 개수 : ', (pred != y_test).sum())
print('값이 1인 사람의 수 : ', (pred == 1).sum())


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))

# testplot = roc_curve(model, x_test, y_test)
from sklearn.metrics import roc_curve

y_score1 = model.predict(x_test)
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)

plt.title('ROC CURVE - Tensor')

plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




"""
# ROC - 준비, 이거 오류남.
pred_label = model.predict(x_test)[:,1]
fprs, tprs, thresholds = roc_curve(y_test, pred_label)
precisions, recalls, thresholds = roc_curve(y_test, pred_label)
plt.figure(figsize = (5,5))

# 대각선
plt.plot([0,1], [0,1], label = 'STR')

# ROC
plt.plot(fprs, tprs, lable = 'ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()
"""