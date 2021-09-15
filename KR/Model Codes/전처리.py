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
import pandas as pd

pd.set_option('max_columns', None)
data = pd.read_csv("../testdata/LoansData.csv") 
# print(data.shape) #252000, 13
# print(data.head(3))
# print(data.info()) 
# print(data['Married/Single'].unique()) #['single' 'married']
# print(data['House_Ownership'].unique()) #['rented' 'norent_noown' 'owned']
# print(data['Car_Ownership'].unique()) #['no' 'yes']
# print(len(data['Profession'].unique())) #51
# print(len(data['CITY'].unique())) #317
# print(len(data['STATE'].unique())) #29

data = data.drop(['Id'], axis = 1) #id칼럼 제거
data['Married/Single'] = data['Married/Single'].map({'single':0, 'married':1}) #['single' 'married'] -> 0,1값으로 대체
data['Car_Ownership'] = data['Car_Ownership'].map({'no':0, 'yes':1}) #['no' 'yes'] -> 0,1값으로 대체

#print(data.head(3))
#print(data[30:35])
#print(data.shape) #252000, 12
print('----------------------------OneHotEncoder---------------------------------------------------------------------------')
# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy as np

data_x = data[['House_Ownership','Profession', 'CITY', 'STATE']]
#print(data_x.head(3))
data_x.loc[:,'House_Ownership'] = LabelEncoder().fit_transform(data_x['House_Ownership'])#['rented':2, 'norent_noown':0, 'owned':1]
data_x.loc[:,'Profession'] = LabelEncoder().fit_transform(data_x['Profession'])
data_x.loc[:,'CITY'] = LabelEncoder().fit_transform(data_x['CITY'])
data_x.loc[:,'STATE'] = LabelEncoder().fit_transform(data_x['STATE'])
# print(data_x.head(3))
# print(data_x['House_Ownership'].unique())
# print(data_x['Profession'].unique())
# print(data_x['CITY'].unique()) 
# print(data_x['STATE'].unique()) 
#---------------------------집여부 원핫인코딩-------------------
housenames=sorted(data['House_Ownership'].unique())
data_x1 = pd.DataFrame(OneHotEncoder().fit_transform(data_x['House_Ownership'].values[:, np.newaxis]).toarray(),\
                        columns=housenames, index = data_x.index)
data_x = pd.concat([data_x, data_x1], axis = 1)
#---------------------------직업 원핫인코딩-------------------
profnames=sorted(data['Profession'].unique())
#print(profnames)
#print(profnames[33])
data_x2 = pd.DataFrame(OneHotEncoder().fit_transform(data_x['Profession'].values[:, np.newaxis]).toarray(),\
                     columns=profnames, index = data_x.index)

data_x = pd.concat([data_x, data_x2], axis = 1)
#print(data_x.head(3))
#-----------------------------도시 원핫인코딩--------------------
citynames=sorted(data['CITY'].unique())
#print(citynames)
data_x3 = pd.DataFrame(OneHotEncoder().fit_transform(data_x['CITY'].values[:, np.newaxis]).toarray(),\
                     columns=citynames, index = data_x.index)

data_x = pd.concat([data_x, data_x3], axis = 1)
#print(data_x.head(3))

#---------------------------주 원핫인코딩-------------------------
statenames=sorted(data['STATE'].unique())
#print(statenames)
data_x4 = pd.DataFrame(OneHotEncoder().fit_transform(data_x['STATE'].values[:, np.newaxis]).toarray(),\
                     columns=statenames, index = data_x.index)

data_x = pd.concat([data_x, data_x4], axis = 1)
#print(data_x.head(3))


data_x = data_x.drop(['House_Ownership','Profession', 'CITY', 'STATE'], axis = 1)
#print(data_x.head(3))
#print(data_x.shape)

#확인용: 첫번째 케이스의 profession은 mechanical_engineer, city는 Rewa, state는 Madhya_Pradesh다.
#print(data[:1][['CITY', 'STATE', 'Profession']])
#원핫인코딩 된 세 변수가 다 1로 지정되었는지 확인
#print(data_x[:1][['Rewa', 'Madhya_Pradesh', 'Mechanical_engineer']])

data_drop = data.drop(['House_Ownership','Profession', 'CITY', 'STATE'], axis = 1)
dataOneHot = pd.concat([data_drop, data_x], axis = 1)
#print(dataOneHot[:1], dataOneHot.shape) #shape은 (252000, 408): 51(직업)+317(도시)+29(주)+3(집여부)+13(원본변수)-5(ID, House, Prof, CITY, STATE)
#print(dataOneHot.head(3))
#print(dataOneHot.info())
print('--------------------------LabelEncoder--------------------------------------------------------------------------')
#LabelEncoder
#print(data.head(3))
# 0 : 'Air_traffic_controller'    1 : 'Analyst'   2 : 'Architect'     3 : 'Army_officer'  4 : 'Artist'
# 5 : 'Aviator'   6 : 'Biomedical_Engineer'   7 : 'Chartered_Accountant'  8 : 'Chef'      9 : 'Chemical_engineer'
# 10 : 'Civil_engineer'   11 : 'Civil_servant'    12 : 'Comedian'     13 : 'Computer_hardware_engineer'
# 14 : 'Computer_operator'    15 : 'Consultant'   16 : 'Dentist'  17:  'Design_Engineer'  18 : 'Designer'
# 19 : 'Drafter'    20 : 'Economist'  21 : 'Engineer'    22 : 'Fashion_Designer'  23 : 'Financial_Analyst'
# 24 : 'Firefighter'  25 : ' Flight_attendant'    26 : 'Geologist'    27 : 'Graphic_Designer'
# 28 : 'Hotel_Manager'    29 : 'Industrial_Engineer'  30 : 'Lawyer'   31 : 'Librarian'    32 : 'Magistrate'
# 33 : 'Mechanical_engineer'  34 : 'Microbiologist'   35 : 'Official' 36 : 'Petroleum_Engineer'   37 : 'Physician'
# 38 : 'Police_officer'   39 : 'Politician'   40 : 'Psychologist' 41 : 'Scientist'    42 : 'Secretary'
# 43 : 'Software_Developer'   44 : 'Statistician' 45 : 'Surgeon'  46 : 'Surveyor' 47 : 'Technical_writer'
# 48 : 'Technician'   49 : 'Technology_specialist'    50 : 'Web_designer'

data_la = data[['House_Ownership','Profession', 'CITY', 'STATE']]
#print(data_la.head(3))
data_la.loc[:,'House_Ownership'] = LabelEncoder().fit_transform(data_la['House_Ownership'])
data_la.loc[:,'Profession'] = LabelEncoder().fit_transform(data_la['Profession'])
data_la.loc[:,'CITY'] = LabelEncoder().fit_transform(data_la['CITY'])
data_la.loc[:,'STATE'] = LabelEncoder().fit_transform(data_la['STATE'])
#print(data_la.head(3))
#print(data_la['Profession'].unique())
#print(data_la['CITY'].unique()) 
#print(data_la['STATE'].unique()) 

dataLabel = pd.concat([data_drop, data_la], axis = 1)
# print(dataLabel.head(3))
# print(dataLabel.info())
# print(dataLabel.shape)

print('--------------------------산업군별------------------------------------------------------------------------------------------')
data_tt = data[['Profession']]
data_tt.loc[:,'Profession'] = LabelEncoder().fit_transform(data_tt['Profession'])
#print(data_tt.head(3))
#print(data_tt['Profession'].unique())

data_tt['Profession'] = data_tt['Profession'].replace([23,15,7], 'Business and Financial Operations Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([47,42,11], 'Office and Administrative Support Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([36,17,29,33,10,2,9,18,46,19,48,21,49,6], 'Architecture and Engineering Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([34,26,1,40,44,41,20], 'Life, Physical, and Social Science Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([43,13,14], 'Computer and Mathematical Occupations')
data_tt['Profession'] = data_tt['Profession'].replace(31, 'Education, Training, and Library Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([32,30,39], 'Legal Occupations')
data_tt['Profession'] = data_tt['Profession'].replace(3, 'Military Specific Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([16,45,37], 'Healthcare Practitioners and Technical Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([12,22,27,50,4], 'Arts, Design, Entertainment, Sports, and Media Occupations')
data_tt['Profession'] = data_tt['Profession'].replace(28, 'Personal Care and Service Occupations')
data_tt['Profession'] = data_tt['Profession'].replace(8, 'Food Preparation and Serving Related Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([0,5,25], 'Transportation and Material Moving Occupations')
data_tt['Profession'] = data_tt['Profession'].replace(35, 'Management Occupations')
data_tt['Profession'] = data_tt['Profession'].replace([24,38], 'Protective Service Occupations')

# print(data_tt['Profession'].unique())
# print(data_tt.info())
# print(data_drop.head(3))

data_drop2 = data.drop(['Profession'], axis = 1)
datapro = pd.concat([data_drop2, data_tt], axis = 1)
#print(datapro.head(3))

print('--------------------------산업군별OneHotEncoder--------------------------------------------------------------------------')
data_xx = datapro[['House_Ownership','Profession', 'CITY', 'STATE']]
#print(data_x.head(3))
data_xx.loc[:,'House_Ownership'] = LabelEncoder().fit_transform(data_xx['House_Ownership'])
data_xx.loc[:,'Profession'] = LabelEncoder().fit_transform(data_xx['Profession'])
data_xx.loc[:,'CITY'] = LabelEncoder().fit_transform(data_xx['CITY'])
data_xx.loc[:,'STATE'] = LabelEncoder().fit_transform(data_xx['STATE'])

#print(data_x.head(3))
# print(data_x['Profession'].unique())
# print(data_x['CITY'].unique()) 
# print(data_x['STATE'].unique()) 

#---------------------------집여부 원핫인코딩-------------------
housenames=sorted(data['House_Ownership'].unique())
#print(housenames)
data_xx1 = pd.DataFrame(OneHotEncoder().fit_transform(data_xx['House_Ownership'].values[:, np.newaxis]).toarray(),\
                        columns=housenames, index = data_xx.index)
data_xx = pd.concat([data_xx, data_xx1], axis = 1)
#---------------------------직업 원핫인코딩-------------------
profnames=sorted(datapro['Profession'].unique())
#print(profnames)
data_xx2 = pd.DataFrame(OneHotEncoder().fit_transform(data_xx['Profession'].values[:, np.newaxis]).toarray(),\
                     columns=profnames, index = data_xx.index)

data_xx = pd.concat([data_xx, data_xx2], axis = 1)
#print(data_xx.head(3))
#-----------------------------도시 원핫인코딩--------------------
citynames=sorted(datapro['CITY'].unique())
#print(citynames)
data_xx3 = pd.DataFrame(OneHotEncoder().fit_transform(data_xx['CITY'].values[:, np.newaxis]).toarray(),\
                     columns=citynames, index = data_xx.index)

data_xx = pd.concat([data_xx, data_xx3], axis = 1)
#print(data_xx.head(3))

#---------------------------주 원핫인코딩-------------------------
statenames=sorted(datapro['STATE'].unique())
#print(statenames)
data_xx4 = pd.DataFrame(OneHotEncoder().fit_transform(data_xx['STATE'].values[:, np.newaxis]).toarray(),\
                     columns=statenames, index = data_xx.index)

data_xx = pd.concat([data_xx, data_xx4], axis = 1)
#print(data_xx.head(3))

data_xx = data_xx.drop(['House_Ownership','Profession', 'CITY', 'STATE'], axis = 1)
#print(data_xx.head(3))

dataOneHot_pro = pd.concat([data_drop, data_xx], axis = 1)
#print(dataOneHot_pro[:1], dataOneHot_pro.shape) #(252000, 370)

print('--------------------------산업군별LabelEncoder--------------------------------------------------------------------------')
data_la2 = datapro[['House_Ownership','Profession', 'CITY', 'STATE']]
#print(data_la2.head(3))
data_la2.loc[:,'House_Ownership'] = LabelEncoder().fit_transform(data_la2['House_Ownership'])
data_la2.loc[:,'Profession'] = LabelEncoder().fit_transform(data_la2['Profession'])
data_la2.loc[:,'CITY'] = LabelEncoder().fit_transform(data_la2['CITY'])
data_la2.loc[:,'STATE'] = LabelEncoder().fit_transform(data_la2['STATE'])
#print(data_la2.head(3))
#print(data_la2['Profession'].unique())
#print(data_la2['CITY'].unique()) 
#print(data_la2['STATE'].unique()) 

dataLabel_pro = pd.concat([data_drop, data_la2], axis = 1)
#print(dataLabel_pro.head(3))
#print(dataLabel_pro.info())

print('------------------------------------------------------------------------------------------------------------------------')
#print(data.shape) #전처리 전 데이터
#print(datapro.shape) #산업군별 전처리 전 데이터
print(dataOneHot.shape) #(252000, 408) OneHot인코딩된 데이터
print(dataLabel.shape) #(252000, 12) Label인코딩된 데이터
print(dataOneHot_pro.shape) #(252000, 372) 산업군별 OneHot인코딩된 데이터
print(dataLabel_pro.shape) #(252000, 12) 산업군별 Label인코딩된 데이터




