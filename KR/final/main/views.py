from django.shortcuts import render
import numpy as np
import pandas as pd
from django.http import HttpResponse
from collections import OrderedDict
# from .파일이름 import 클래스명
# from .loan_data import loan_data # 어떤 분들은 . <- 없이 해야지 불러 옴
import os

# f = 클래스명()
# f.함수명()
# 어딘가 메모리에 리턴 값들이 저장됨
# f = loan_data() # 외부 라이브러리의 클래스 저장

datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            '../final/main/static/need_data.csv')
data = pd.read_csv(datas)


# Create your views here.
# 메인 페이지
def MainFunc(request):
    
    return render(request,'main.html')


# 팀 소개 페이지
def AboutUs(request):
    return render(request, 'about.html')

# 그래프 페이지
def DataRecap(request):
    
    dict = {}
    for i in range(len(data['columns'])):
        dict[str(data['columns'][i])] = data['values'][i]
    print(dict.items())
    
    return render(request, 'data.html', dict)


def ListModels(request):
    return render(request, 'model.html')

def ModelPred(request):
    if request.method=='GET':
        return render(request, 'predict.html')
    elif request.method=='POST':
        try:
            import pandas as pd
            import pickle
            import numpy as np
            import json
            from django.http.response import HttpResponse
            import os
            from sklearn.preprocessing import StandardScaler
            import re
            # income(소득)    Income of the user(이용자의 소득)   
            # age(연령)    Age of the user(사용자의 연령)    
            # experience(경험)    Professional experience of the user in years(년의 사용자 전문 경험)   
            # profession(직업)    Profession(직업)   
            # married(결혼한)    Whether married or single(결혼여부 또는 독신)    
            # house_ownership    Owned or rented or neither(소유 또는 임대 어느 쪽도)   
            # car_ownership    Does the person own a car(사람이 자동차를 소유합니까?)   
            # risk_flag    Defaulted on a loan(대출 채무 불이행)   
            # currentjobyears(현재 작업연도)    Years of experience in the current job(현재 직업에서의 경험의 년) 
            # currenthouseyears(현재 주택년)    Number of years in the current residence(현재 거주지의 연도 수)         
            
            #-----------------------1단계: 입력한 데이터 전처리-------------------------
            pred_samp=np.zeros([1,25])
            columns=['Income', 'Age', 'Experience', 'Married/Single', 'Car_Ownership',
                   'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'rented',
                   'norent_noown', 'owned', 'Architecture and Engineering Occupations',
                   'Arts, Design, Entertainment, Sports, and Media Occupations',
                   'Business and Financial Operations Occupations',
                   'Computer and Mathematical Occupations',
                   'Education, Training, and Library Occupations',
                   'Food Preparation and Serving Related Occupations',
                   'Healthcare Practitioners and Technical Occupations',
                   'Legal Occupations', 'Life, Physical, and Social Science Occupations',
                   'Management Occupations', 'Military Specific Occupations',
                   'Office and Administrative Support Occupations',
                   'Personal Care and Service Occupations',
                   'Protective Service Occupations',
                   'Transportation and Material Moving Occupations']
            pred_samp=pd.DataFrame(pred_samp, columns=columns, index=None)
            
            #데이터 인식
            #한국 원화로 받을 경우 .063을 곱한다
            pred_samp[['Income']]=round(float(request.POST.get('Income'))*0.063)
            pred_samp[['Age']]=request.POST.get('Age')
            pred_samp[['Experience']]=request.POST.get('experience')
            pred_samp[['Married/Single']]=request.POST.get('married')
            pred_samp[['Car_Ownership']]=request.POST.get('Car_Ownership')
            pred_samp[['CURRENT_JOB_YRS']]=request.POST.get('CURRENT_JOB_YRS')
            pred_samp[['CURRENT_HOUSE_YRS']]=request.POST.get('CURRENT_HOUSE_YRS')
            pred_samp[[request.POST.get('Profession')]]=1
            pred_samp[[request.POST.get('house_ownership')]]=1
            pd.set_option('max_columns', None)
            print(pred_samp) #인풋 확인용
            
            #------------------------2단계: 모델로 예측---------------------
            #모델 경로 지정
            #모델들은 전부 쟝고 시작폴더 (final) 안에 있다.
            xgtrainpath=os.path.join(os.path.dirname(os.path.dirname(__file__)),'lgbmpred_train.csv')
            rfpath=os.path.join(os.path.dirname(os.path.dirname(__file__)),'rfmodel_simple_n70.dat')
            xgpath=os.path.join(os.path.dirname(os.path.dirname(__file__)),'lgbmodel_simple.dat')
            tfpath=os.path.join(os.path.dirname(os.path.dirname(__file__)),'tensor_simple.hdf5')
    
            #모델 부르기
            model=pickle.load(open(rfpath, 'rb'))
            model2=pickle.load(open(xgpath, 'rb'))
            from tensorflow.keras.models import load_model
            model3=load_model(tfpath)
            
            #--------------예측, 1:Random Forest, 2:LGBM, 3:TensorFlow--------
            pred_val1=model.predict(pred_samp)[0]
            #print(pred_val1)
            if pred_val1==0:
                output1='채무불이행 위험군이 아닙니다'
            elif pred_val1==1:
                output1='채무불이행 위험군입니다'
            
            #LGBM모델은 standard scaler와 변수들의 정규표현식 필터를 해야한다
            lgbm_train = pd.read_csv(xgtrainpath) 
            pred_samp_lgbm = pred_samp.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            lgbm_train = lgbm_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            
            #표준화
            #표준화를 위한 기본 데이터 train을 가져온다
            scaler = StandardScaler()
            train_sc = scaler.fit_transform(lgbm_train)
            pred_samp_sc = scaler.transform(pred_samp_lgbm)
            pred_val2=model2.predict(pred_samp_sc)[0]
            #print(pred_val2)
            if pred_val2==0:
                output2='채무불이행 위험군이 아닙니다'
            elif pred_val2==1:
                output2='채무불이행 위험군입니다'
            
            
            #TF모델은 array로 전환 후 예측해야한다.
            pred_samp=np.asarray(pred_samp).astype(np.float32)
            pred_val3=model3.predict(pred_samp).item(0)
            #print(pred_val3)
            if pred_val3<0.5:
                output3='채무불이행 위험군이 아닙니다'
            elif pred_val3>0.5:
                output3='채무불이행 위험군입니다'
            #TF는 0과 1이 아닌 확률값으로 주기 때문에 %로 변환 후 쟝고에 표출하기 위해 사용한다
            pred_val3=np.round(pred_val3, decimals=4)*100
            #print(pred_val3)
            
            #전체 예측결과 값 표출
            #print(output1, output2, output3)
            
            #----------------------3단계: 결과 HttpResponse로 반환------------
            #쟝고로 보낼 변수들을 dict타입으로 저장한다
            outputs={'result_rf':output1, 'result_xg':output2, 'result_tf': output3, 'coef_tf':pred_val3}
            #print(outputs) #확인용
            
            #Ajax 방식으로 들어왔기 때문에 HttpReponse를 통해 json파일 타입으로 보낸다
            return HttpResponse(json.dumps(outputs), content_type='application/json')
        except Exception as e:
            print('1차: post 인식 처리 후 오류')
            print(e)
            return render(request, 'predict.html')



# 테스트용 페이지 입니다.

def Test(request):
    dict = {}
    for i in range(len(data['columns'])):
        dict[str(data['columns'][i])] = data['values'][i]
    # print(dict.items())
    
    return render(request, 'test.html', dict)





# 모델 분석 결과 페이지
def model_DecisionTree(request):
    return render(request, 'model_DecisionTree.html')
def model_KNN(request):
    return render(request, 'model_KNN.html')
def model_MLP(request):
    return render(request, 'model_MLP.html')
def model_SVM(request):
    return render(request, 'model_SVM.html')
def model_Tensor(request):
    return render(request, 'model_Tensor.html')
def model_LGBMClassifier(request):
    return render(request, 'model_LGBMClassifier.html')
def model_LogisticRegression(request):
    return render(request, 'model_LogisticRegression.html')
def model_NaiveBayes(request):
    return render(request, 'model_NaiveBayes.html')
def model_RandomForest(request):
    return render(request, 'model_RandomForest.html')
def model_XGBClassifier(request):
    return render(request, 'model_XGBClassifier.html')


