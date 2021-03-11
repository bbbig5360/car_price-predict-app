import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg') # 서버에서, 화면에 표시하기 위해서 필요하다.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import h5py
import joblib
import pickle

def run_ml_app():
    st.subheader('ML 화면입니다.')

    model = tensorflow.keras.models.load_model('data/car_ai.h5')
        
    gender = st.radio('성별을 선택하시오', ['남자','여자'])
    if gender =='남자':
        gen_num = 1
    if gender =='여자':
        gen_num = 0

    age = st.number_input('나이를 입력하세요',0,100)
    salary = st.number_input('연봉을 입력하세요(달러)', min_value=0)
    debt = st.number_input('카드빚을 입력하세요(달러)', min_value=0)
    worth = st.number_input('순 자산을 입력하세요(달러)', min_value=0)

    
    new_data = np.array([gen_num, age, salary, debt, worth])

    # 데이터프레임으로 스케일링했기때문에 차원을 맞춰준다.(2차원으로)
    new_data = new_data.reshape(1,-1)

    # if st.button('입력한 값을 보여줍니다'):
    #     st.subheader('입력한 값을 보여줍니다')
    #     st.write(input_list)
    #     st.write(type(input_list))

    if st.button('고객이 살 차량의 금액을 예측합니다'):
        sc_X = joblib.load('data/sc_X.pkl')
        new_data = sc_X.transform(new_data)

        y_pred = model.predict(new_data)
        sc_y = joblib.load('data/sc_y.pkl')

        # 금액도 스케일링해서 학습했었기 떄문에 끝난 후 역변환해준다.
        y_pred = sc_y.inverse_transform(y_pred)

        #st.write(y_pred)
        st.write('고객은 {:,.0f}달러의 차를 살 것입니다'.format(y_pred[0][0], ))

