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
import os

class loan_data():
    def __init__(self): 
        #--------------------------원본 데이터----------------------------
        #경로설정필수
        # data_base = pd.read_csv("LoansData.csv") 
        # data_base = pd.read_csv("C:/Users/Jo/OneDrive/Desktop/Training Data.csv") 
        
        
        datas = os.path.join(os.path.dirname(os.path.dirname(__file__)),
             '../final/main/static/Training Data.csv')
        
        data_base = pd.read_csv(datas)
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
        datas1 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            '../final/main/static/data_onehot_one.csv')
        # 데이터 불러오기
        # data_onehot_one = pd.read_csv(datas1)
        data_onehot_one=data_label_one.copy()
        
        
        # """
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
        
        data_onehot_one = data_onehot_one.drop(['Profession', 'CITY', 'STATE', 'House_Ownership'], axis = 1)
        
        #함수호출용 self버전 지정
        self.data_onehot_one=data_onehot_one
        #shape은 (252000, 406): 51(직업)+317(도시)+29(주)+13(원본변수)-4(ID, Prof, CITY, STATE)
        
        
        # 데이터 저장
        self.data_onehot_one.to_csv('data_onehot_one.csv', sep = ',', index = False)
        # """
        
        #--------------------------산업군 코딩--------------------------------------------------------
        
        # """
        #산업군 수:15개
        prof_category=data_label_one.copy()[['Profession']]
        #산업군 별로 분류
        prof_category['Prof_Category']=prof_category['Profession']
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([23,15,7], 'Business_Financial')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([47,42,11], 'Office_Administrative_Support')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([36,17,29,33,10,2,9,18,46,19,48,21,49,6], 'Architecture_Engineering')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([34,26,1,40,44,41,20], 'Life_Physical_Social_Science')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([43,13,14], 'Computer_Mathematical')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(31, 'Education_Training_Library')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([32,30,39], 'Legal')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(3, 'Military_Specific')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([16,45,37], 'Healthcare_Practitioners_Technical')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([12,22,27,50,4], 'Arts_Design_Entertainment_Sports_Media')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(28, 'Personal_Care_Service')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(8, 'Food_Preparation_Serving_Related')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([0,5,25], 'Transportation_Material_Moving')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace(35, 'Management')
        prof_category['Prof_Category'] = prof_category['Prof_Category'].replace([24,38], 'Protective_Service')
        prof_category = prof_category.drop(['Profession'], axis = 1)
        
        #------------------------------------Full Data--------------------------------------------
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
        
        
        # """
        
        
        # ------------------------- 추가된 내용 ----------------------
        
        # 데이터 저장
        # data_onehot_two.to_csv('data_onehot_two.csv', sep = ',', index = False) # 컬럼 확인용
        
        # 데이터 불러오기
        datas2 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
             '../final/main/static/data_onehot_two.csv')
        
        self.data_onehot_two = pd.read_csv(datas2)
        # print(self.data_onehot_two)
        # 각 변수 가져오기(그래프 요소)

        
        # 수입
        # print(data_onehot_two['Income'].describe().astype('int'))
        
        # count     252000
        # mean     4997116
        # std      2878311
        # min        10310
        # 25%      2503015
        # 50%      5000694
        # 75%      7477502
        # max      9999938
        
        self.Income_top = [] # min ~ 25% 미만
        self.Income_mid = [] # 25% ~ 75% 미만
        self.Income_bot = [] # 75 ~ max이하
        
        # print(self.data_onehot_two['Income'][0])
        for i in range(0, 252000):
            if self.data_onehot_two['Income'][i] < 2503015:
                self.Income_bot.append(self.data_onehot_two['Income'][i])
            elif self.data_onehot_two['Income'][i] < 7477502:
                self.Income_mid.append(self.data_onehot_two['Income'][i])
            elif self.data_onehot_two['Income'][i] >= 7477502:
                self.Income_top.append(self.data_onehot_two['Income'][i])
        self.Income_top_mean = int(sum(self.Income_top) / len(self.Income_top)) 
        self.Income_mid_mean = int(sum(self.Income_mid) / len(self.Income_mid)) 
        self.Income_bot_mean = int(sum(self.Income_bot) / len(self.Income_bot)) 
        
        
        # 연령 -> 10대 20대 30대 ... 데이터 나누기
        # print(data_onehot_two['Age'].describe().astype('int'))
        
        # count    252000
        # mean         49
        # std          17
        # min          21
        # 25%          35
        # 50%          50
        # 75%          65
        # max          79
       
        self.Age_20s = []; self.Age_30s = []; self.Age_40s = []; self.Age_50s = []; self.Age_60s = []; self.Age_70s = []
        
        
        for i in range(0, 252000):
            if self.data_onehot_two['Age'][i] < 30:
                self.Age_20s.append(self.data_onehot_two['Age'][i])
            elif self.data_onehot_two['Age'][i] < 40:
                self.Age_30s.append(self.data_onehot_two['Age'][i])
            elif self.data_onehot_two['Age'][i] < 50:
                self.Age_40s.append(self.data_onehot_two['Age'][i])
            elif self.data_onehot_two['Age'][i] < 60:
                self.Age_50s.append(self.data_onehot_two['Age'][i])
            elif self.data_onehot_two['Age'][i] < 70:
                self.Age_60s.append(self.data_onehot_two['Age'][i])
            elif self.data_onehot_two['Age'][i] >= 70:
                self.Age_70s.append(self.data_onehot_two['Age'][i])

        self.Age_name = ['20대', '30대', '40대', '50대', '60대', '70대']        
        self.Age_value = [len(self.Age_20s), len(self.Age_30s), len(self.Age_40s), len(self.Age_50s),
                       len(self.Age_60s), len(self.Age_70s)]
        
        
        # 결혼 여부
        self.Married = self.data_onehot_two[self.data_onehot_two['Married/Single'] == 0].count()[0] # 
        self.Single = self.data_onehot_two[self.data_onehot_two['Married/Single'] == 1].count()[0] # 
        
        self.Marital_status = ['Married', 'Single']
        # print(self.Married); print(self.Single)
        
        
        # 주 - 상위 5개
        STATE_name = []
        STATE_value = []
        self.STATE_Top5_name = []
        self.STATE_Top5_value = []
        
        for i in range(377, 406):
            STATE_name.append(self.data_onehot_two.columns[i])
            STATE_value.append(self.data_onehot_two.iloc[:, i].sum().astype('int'))
        
        STATE_dict = dict(zip(STATE_name, STATE_value))
        STATE_dict_sort = sorted(STATE_dict.items(), key = lambda x: x[1], reverse = True)
        self.STATE_Top5_name.extend(list(STATE_dict_sort[i][0] for i in range(5)))
        self.STATE_Top5_value.extend(list(STATE_dict_sort[i][1] for i in range(5)))


        # 도시 - 상위 10개
        CITY_name = []
        CITY_value = []
        self.CITY_Top10_name = []
        self.CITY_Top10_value = []
        
        for i in range(60, 377):
            CITY_name.append(self.data_onehot_two.columns[i])
            CITY_value.append(self.data_onehot_two.iloc[:, i].sum().astype('int'))
        
        CITY_dict = dict(zip(CITY_name, CITY_value))
        CITY_dict_sort = sorted(CITY_dict.items(), key = lambda x: x[1], reverse = True)
        self.CITY_Top10_name.extend(list(CITY_dict_sort[i][0] for i in range(10)))
        self.CITY_Top10_value.extend(list(CITY_dict_sort[i][1] for i in range(10)))
        
        
        for i in range(len(self.CITY_Top10_name)):
            a = self.CITY_Top10_name[i].replace("[", ("_"))
            a = a.replace("]", "")
            self.CITY_Top10_name[i] = a
        # print(self.CITY_Top10_name)
        
        
        # 집 소유
        self.House_Ownership_0 = self.data_onehot_two[self.data_onehot_two['House_Ownership'] == 0].count()[0] # rented : 231898
        self.House_Ownership_1 = self.data_onehot_two[self.data_onehot_two['House_Ownership'] == 1].count()[0] # norent_noown : 7184
        self.House_Ownership_2 = self.data_onehot_two[self.data_onehot_two['House_Ownership'] == 2].count()[0] # owned : 12918
        
        
        # 자동차 소유
        self.Car_Ownership_0 = self.data_onehot_two[self.data_onehot_two['Car_Ownership'] == 0].count()[0] # no : 176000
        self.Car_Ownership_1 = self.data_onehot_two[self.data_onehot_two['Car_Ownership'] == 1].count()[0] # yes : 76000
        
        
        # 경력
        self.Experience_1 = [] # 0 ~ 5
        self.Experience_2 = [] # 5 ~ 10
        self.Experience_3 = [] # 10 ~ 15
        self.Experience_4 = [] # 15 ~ 20
        
        
        
        for i in range(0, 252000):
            if self.data_onehot_two['Experience'][i] <= 5:
                self.Experience_1.append(self.data_onehot_two['Experience'][i])
            elif self.data_onehot_two['Experience'][i] <= 10:
                self.Experience_2.append(self.data_onehot_two['Experience'][i])
            elif self.data_onehot_two['Experience'][i] <= 15:
                self.Experience_3.append(self.data_onehot_two['Experience'][i])
            elif self.data_onehot_two['Experience'][i] <= 20:
                self.Experience_4.append(self.data_onehot_two['Experience'][i])
        
        self.Experience_name = ['Exp_1', 'Exp_2', 'Exp_3', 'Exp_4']        
        self.Experience_value = [len(self.Experience_1), len(self.Experience_2), len(self.Experience_3), len(self.Experience_4)]
        
        
        
        
        # 직업(대분류)
        self.prof_category_name = []
        self.prof_category_data = []
        max_len = len(self.data_onehot_two.columns) # 421     
        
        
        for i in range(406, max_len):
            self.prof_category_name.append(self.data_onehot_two.columns[i])
            self.prof_category_data.append(self.data_onehot_two.iloc[:, i].sum().astype('int'))
        print(self.data_onehot_two.columns[406:])
        # 직업 데이터 입력 순서
        # Architecture and Engineering Occupations,
        # Arts, Design, Entertainment, Sports, and Media Occupations,
        # Business and Financial Operations Occupations,
        # Computer and Mathematical Occupations,
        # Education, Training, and Library Occupations,
        # Food Preparation and Serving Related Occupations,
        # Healthcare Practitioners and Technical Occupations,
        # Legal Occupations, 
        # Life, Physical, and Social Science Occupations,
        # Management Occupations, 
        # Military Specific Occupations,
        # Office and Administrative Support Occupations,
        # Personal Care and Service Occupations,
        # Protective Service Occupations,
        # Transportation and Material Moving Occupations
        
        
        
        # 근속년수 : 0 ~ 14
        self.Cur_Job_name = ['Cur_Job_1', 'Cur_Job_2', 'Cur_Job_3']
        self.Cur_Job_data1 = []
        self.Cur_Job_data2 = []
        self.Cur_Job_data3 = []
        
        for i in range(0, 252000):
            if self.data_onehot_two['CURRENT_JOB_YRS'][i] <= 5:
                self.Cur_Job_data1.append(self.data_onehot_two['CURRENT_JOB_YRS'][i])
            elif self.data_onehot_two['CURRENT_JOB_YRS'][i] <= 10:
                self.Cur_Job_data2.append(self.data_onehot_two['CURRENT_JOB_YRS'][i])
            elif self.data_onehot_two['CURRENT_JOB_YRS'][i] <= 15:
                self.Cur_Job_data3.append(self.data_onehot_two['CURRENT_JOB_YRS'][i])
        
        self.Cur_Job_value = [len(self.Cur_Job_data1), len(self.Cur_Job_data2), len(self.Cur_Job_data3)]
        
        
        # 주거년수 10, 11, 12, 13, 14
        self.Cur_House_name = ['Cur_House_1', 'Cur_House_2', 'Cur_House_3', 'Cur_House_4', 'Cur_House_5']
        self.Cur_House_data1 = []
        self.Cur_House_data2 = []
        self.Cur_House_data3 = []
        self.Cur_House_data4 = []
        self.Cur_House_data5 = []
        
        for i in range(0, 252000):
            if self.data_onehot_two['CURRENT_HOUSE_YRS'][i] == 10:
                self.Cur_House_data1.append(self.data_onehot_two['CURRENT_HOUSE_YRS'][i])
            elif self.data_onehot_two['CURRENT_HOUSE_YRS'][i] == 11:
                self.Cur_House_data2.append(self.data_onehot_two['CURRENT_HOUSE_YRS'][i])
            elif self.data_onehot_two['CURRENT_HOUSE_YRS'][i] == 12:
                self.Cur_House_data3.append(self.data_onehot_two['CURRENT_HOUSE_YRS'][i])
            elif self.data_onehot_two['CURRENT_HOUSE_YRS'][i] == 13:
                self.Cur_House_data4.append(self.data_onehot_two['CURRENT_HOUSE_YRS'][i])
            elif self.data_onehot_two['CURRENT_HOUSE_YRS'][i] == 14:
                self.Cur_House_data5.append(self.data_onehot_two['CURRENT_HOUSE_YRS'][i])
                
                
        self.Cur_House_value = [len(self.Cur_House_data1), len(self.Cur_House_data2), len(self.Cur_House_data3), len(self.Cur_House_data4), len(self.Cur_House_data5)]
        
        
        
        
        
        # 위험도
        # Risk_Flag_0 = (self.data_onehot_two['Risk_Flag'].isin([0])).count()
        self.Risk_Flag_0 = self.data_onehot_two[self.data_onehot_two['Risk_Flag'] == 0].count()[0] # safe : 221004
        self.Risk_Flag_1 = self.data_onehot_two[self.data_onehot_two['Risk_Flag'] == 1].count()[0] # warning : 30996
        
        
        # data_onehot_one = pd.concat([data_onehot_one, df_profession], axis = 1)
        need_data = pd.DataFrame()

        need_data['columns'] = ['Income_top', 'Income_mid', 'Income_bot'] + self.Age_name + self.Marital_status + self.CITY_Top10_name + self.STATE_Top5_name +\
                                ['rented', 'norent_noown', 'owned'] + ['car_no', 'car_yes'] + self.Experience_name + self.prof_category_name +\
                                self.Cur_Job_name + self.Cur_House_name + ['Risk_Flag_safe', 'Risk_Flag_warning'] 
        
        need_data['values'] = [self.Income_top_mean, self.Income_mid_mean, self.Income_bot_mean] +\
                                self.Age_value + [self.Married, self.Single] + self.CITY_Top10_value + self.STATE_Top5_value + [self.House_Ownership_0, self.House_Ownership_1, self.House_Ownership_2] +\
                                [self.Car_Ownership_0, self.Car_Ownership_1] + self.Experience_value + self.prof_category_data +\
                                self.Cur_Job_value + self.Cur_House_value + [self.Risk_Flag_0, self.Risk_Flag_1]
        
        # 그래프 데이터 저장
        need_data.to_csv('need_data.csv', sep = ',', index = False)
        
        
        # ---------------------------------------------------------

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
    
    # def Need_Graph(self):
    # # 그래프 변수들 반환
    #     return self.Income_top_mean, self.Income_mid_mean, self.Income_bot_mean,\
    #         self.Age_20s, self.Age_30s, self.Age_40s, self.Age_50s, self.Age_60s, self.Age_70s,\
    #         self.STATE_Top5_name, self.STATE_Top5_value,\
    #         self.CITY_Top10_name, self.CITY_Top10_value,\
    #         self.House_Ownership_0, self.House_Ownership_1, self.House_Ownership_2,\
    #         self.Car_Ownership_0, self.Car_Ownership_1,\
    #         self.Experience_1, self.Experience_2, self.Experience_3, self.Experience_4,\
    #         self.prof_category_name, self.prof_category_data,\
    #         self.Risk_Flag_0, self.Risk_Flag_1

#확인용

test=loan_data()
# b1=test.basedata()
# b2=test.basedata_labelencode()
# b3=test.basedata_onehotencode()
# b4=test.fulldata()
# b5=test.fulldata_labelencode()
# b6=test.fulldata_onehotencode()
# b7=test.Need_Graph()


# print(b1.shape, b2.shape, b3.shape, b4.shape, b5.shape, b6.shape)
#(252000, 13) (252000, 12) (252000, 406) (252000, 14) (252000, 13) (252000, 421)
# pd.set_option('display.max_columns', None)
# print(b5.head(3))
# print(b6.head(3))
