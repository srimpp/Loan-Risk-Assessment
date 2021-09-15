#-----------------------데이터 종류, 호출 함수별로 목록---------------------------------
# basedata() : 원본 데이터, shape:(252000, 13)
# basedata_labelencode() : 원본데이터 + 라벨인코딩, ID제거, shape: (252000, 12)
# basedata_onehotencode() : 원본데이터 + 원핫인코딩, ID제거, shape: (252000, 406)
# fulldata() : 원본 데이터 + 산업군 변수, shape: (252000, 14)
# fulldata_labelencode() : 원본 데이터 + 산업군 변수 + 라벨인코딩, ID제거, shape: (252000, 13)
# fulldata_onehotencode() : 원본 데이터 + 산업군 변수 + 원핫인코딩, ID제거, shape: (252000, 421)

#------------------------변수설명-----------------------------------
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
# city(도시)    City of residence(레지던스 시티)    
# state(주)    State of residence(거주 상태)

#확인용: 첫번째 케이스의 profession은 mechanical_engineer, city는 Rewa, state는 Madhya_Pradesh다.
 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy as np

#file 불러오고 import loan_data하면 실행가능
class loan_data():
    def __init__(self): 
        #--------------------------원본 데이터----------------------------
        #경로설정필수
        data_base = pd.read_csv("LoansData.csv") 
        #함수호출용 self버전 지정
        self.data_base=data_base
        
        #결혼여부, 집여부, 차여부 수치화
        data_map=data_base.copy()
        data_map['Married/Single'] = data_map['Married/Single'].map({'single':0, 'married':1}) #['single' 'married'] -> 0,1값으로 대체
        data_map['House_Ownership'] = data_map['House_Ownership'].map({'rented':0, 'norent_noown':1, 'owned':2}) #['rented' 'norent_noown' 'owned'] -> 0,1,2대체
        data_map['Car_Ownership'] = data_map['Car_Ownership'].map({'no':0, 'yes':1}) #['single' 'married'] -> 0,1값으로 대체
        
        #ID제거
        data_map = data_map.drop(['Id'], axis = 1)
        
        #-------------------------LabelEncoding---------------------------------------------------------------------------
        #직업, 도시, 주 라벨화, 라벨화는 자동으로 알파벳순으로 나열한다
        data_label_one=data_map.copy()
        data_label_one.loc[:,'Profession'] = LabelEncoder().fit_transform(data_label_one['Profession'])
        data_label_one.loc[:,'CITY'] = LabelEncoder().fit_transform(data_label_one['CITY'])
        data_label_one.loc[:,'STATE'] = LabelEncoder().fit_transform(data_label_one['STATE'])
        #함수호출용 self버전 지정
        self.data_label_one=data_label_one
        
        #-----------------------------OneHotEncoding-----------------------------------------
        data_onehot_one=data_label_one.copy()
        #집여부 원핫인코딩
        housenames=['rented','norent_noown','owned']
        df_house = pd.DataFrame(OneHotEncoder().fit_transform(data_onehot_one['House_Ownership'].values[:, np.newaxis]).toarray(),\
                               columns=housenames, index = data_onehot_one.index)
        data_onehot_one = pd.concat([data_onehot_one, df_house], axis = 1)
        
        
        #직업 원핫인코딩
        profnames=sorted(data_base['Profession'].unique())
        df_profession = pd.DataFrame(OneHotEncoder().fit_transform(data_onehot_one['Profession'].values[:, np.newaxis]).toarray(),\
                               columns=profnames, index = data_onehot_one.index)
        data_onehot_one = pd.concat([data_onehot_one, df_profession], axis = 1)
        
        #도시 원핫인코딩
        citynames=sorted(data_base['CITY'].unique())
        df_cities = pd.DataFrame(OneHotEncoder().fit_transform(data_onehot_one['CITY'].values[:, np.newaxis]).toarray(),\
                               columns=citynames, index = data_onehot_one.index)
        data_onehot_one = pd.concat([data_onehot_one, df_cities], axis = 1)
        
        #주 원핫인코딩
        statenames=sorted(data_base['STATE'].unique())
        df_state = pd.DataFrame(OneHotEncoder().fit_transform(data_onehot_one['STATE'].values[:, np.newaxis]).toarray(),\
                               columns=statenames, index = data_onehot_one.index)
        data_onehot_one = pd.concat([data_onehot_one, df_state], axis = 1)
        data_onehot_one = data_onehot_one.drop(['House_Ownership', 'Profession', 'CITY', 'STATE'], axis = 1)
        #함수호출용 self버전 지정
        self.data_onehot_one=data_onehot_one
        #shape은 (252000, 406): 51(직업)+317(도시)+29(주)+13(원본변수)+3(집)-5(ID, Prof, CITY, STATE), updated shape:408
        
        #--------------------------산업군 코딩--------------------------------------------------------
        #산업군 수:15개
        prof_category=data_label_one.copy()[['Profession']]
        #산업군 별로 분류
        prof_category['Prof_Category']=prof_category['Profession']
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([23,15,7], 'Business and Financial Operations Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([47,42,11], 'Office and Administrative Support Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([36,17,29,33,10,2,9,18,46,19,48,21,49,6], 'Architecture and Engineering Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([34,26,1,40,44,41,20], 'Life, Physical, and Social Science Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([43,13,14], 'Computer and Mathematical Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(31, 'Education, Training, and Library Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([32,30,39], 'Legal Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(3, 'Military Specific Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([16,45,37], 'Healthcare Practitioners and Technical Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([12,22,27,50,4], 'Arts, Design, Entertainment, Sports, and Media Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(28, 'Personal Care and Service Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(8, 'Food Preparation and Serving Related Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([0,5,25], 'Transportation and Material Moving Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(35, 'Management Occupations')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([24,38], 'Protective Service Occupations')
        prof_category = prof_category.drop(['Profession'], axis = 1)
        
        #------------------------------------Full Data--------------------------------------
        data_full=data_base.copy()
        data_full=pd.concat([data_full, prof_category], axis = 1)
        #함수호출용 self버전 지정
        self.data_full=data_full
        
        #----------------------------------라벨인코딩 Full Data---------------------------------------
        #원핫인코딩에 사용될 산업군 이름 목록
        prof_category_names=sorted(prof_category['Prof_Category'].unique())
        #라벨인코딩
        prof_category.loc[:,'Prof_Category']=LabelEncoder().fit_transform(prof_category['Prof_Category'])
        data_label_two=data_label_one.copy()
        data_label_two=pd.concat([data_label_two, prof_category], axis = 1)
        #함수호출용 self버전 지정
        self.data_label_two=data_label_two
        
        #----------------------------------원핫인코딩 full data-------------------------------------
        prof_category = pd.DataFrame(OneHotEncoder().fit_transform(prof_category['Prof_Category'].values[:, np.newaxis]).toarray(),\
                                columns=prof_category_names, index = prof_category.index)
        data_onehot_two=data_onehot_one.copy()
        data_onehot_two=pd.concat([data_onehot_two, prof_category], axis = 1)
        #함수호출용 self버전 지정
        self.data_onehot_two=data_onehot_two
        
        #data_onehot_one.to_csv('basedata_onehot2.csv',sep=',', index=False, encoding='utf-8-sig')

        
        
        #-----------------------------------산업군 ONLY 데이터--------------------------------------
        data_career=data_map.copy()
        #집여부 원핫인코딩
        housenames=['rented','norent_noown','owned']
        data_career = pd.concat([data_career, df_house], axis = 1)
        data_career = pd.concat([data_career, prof_category], axis = 1)
        data_career = data_career.drop(['House_Ownership', 'Profession', 'CITY', 'STATE'], axis = 1)
        data_career.to_csv('simpledata_career.csv',sep=',', index=False, encoding='utf-8-sig')
        self.data_career=data_career
        

    def basedata(self):
    #아무것도 없는 기본데이터
        return self.data_base
    
    def basedata_labelencode(self):
    #산업군 미포함 라벨데이터
        return self.data_label_one
    
    def basedata_onehotencode(self):
    #산업군 미포함 원핫데이터
        return self.data_onehot_one
    
    def fulldata(self):
    #산업군 포함한 기본데이터
        return self.data_full
    
    def fulldata_labelencode(self):
    #산업군 포함한 라벨인코드
        return self.data_label_two

    def fulldata_onehotencode(self):
    #산업군 포함한 원핫인코드
        return self.data_onehot_two
    
    def simpledata_career(self):
    #산업군 포함, 직업, 도시, 주 제외 원핫인코드
        return self.data_career

#확인용
test=loan_data()
# b1=test.basedata()
# b2=test.basedata_labelencode()
#b3=test.basedata_onehotencode()
# b4=test.fulldata()
# b5=test.fulldata_labelencode()
# b6=test.fulldata_onehotencode()
b7=test.simpledata_career()
# print(b1.shape, b2.shape, b3.shape, b4.shape, b5.shape, b6.shape)
#(252000, 13) (252000, 12) (252000, 406) (252000, 14) (252000, 13) (252000, 421)
pd.set_option('display.max_columns', None)
# print(b5.head(3))
# print(b6.head(3))
print(b7.shape)
print(b7.head(3))

#base 13+3(House)+15(Career)-5(ID, STATE, CITY, Profession, House)=26 
#(252000, 26)
