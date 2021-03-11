import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg') # 서버에서, 화면에 표시하기 위해서 필요하다.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import h5py




def run_eda_app():
    st.subheader('EDA 화면입니다.')

    radio_menu = ['데이터 프레임', '통계치']
    selected_radio = st.radio('선택하세요', radio_menu)

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1' )
    
    if selected_radio == '데이터 프레임':
        st.dataframe(car_df)

    if selected_radio == '통계치':
        st.dataframe(car_df.describe())
    
    st.write('')
    st.write('')

    data_col = car_df.columns
    sel_col = st.multiselect('보고싶은 내용들을 고르시오', data_col)
    
    if sel_col :
        st.dataframe(car_df[sel_col])
    else :
        st.write('선택한 컬럼이 없습니다.')
    st.write('')
    st.write('')



    corr_col = car_df.columns[ car_df.dtypes != object ]
    sel_corr = st.multiselect('보고싶은 상관계수의 데이터를 고르시오', corr_col)

    if sel_corr :
        st.dataframe(car_df[sel_corr].corr())
        fig1 = sns.pairplot(data = car_df[ sel_corr ])
        st.pyplot(fig1)
        
    else :
        st.write('선택한 컬럼이 없습니다.')

    st.write('')
    st.write('')
    st.write('')

    Mm_col = car_df.columns[ car_df.dtypes != object ]
    sel_Mm = st.selectbox('보고싶은 내용들을 고르시오', Mm_col)
    
    if sel_Mm : 
        st.write('최대값은 {}이고, 최소값은 {}입니다'.format(car_df[sel_Mm].min(), car_df[sel_Mm].max()))
        st.write('최소값인 사람의 데이터')
        st.dataframe(car_df.loc[car_df[sel_Mm]==car_df[sel_Mm].min(),])
        st.write('최대값인 사람의 데이터')
        st.dataframe(car_df.loc[car_df[sel_Mm]==car_df[sel_Mm].max(),])
    else:
        st.write('선택한 컬럼이 없습니다.')

    
    #이름을 찾아서 내용들을 찾는 법

    input_text = st.text_input('찾으시려는 이름을 입력하시오')
    
    result = car_df.loc[ car_df['Customer Name'].str.contains(input_text, case=False), ]
    st.dataframe(result)

    # if input_text:
    #     for name in car_df['Customer Name'].values:
    #         if input_text.lower() in name.lower() :
    #             st.write(name)
    #             st.write(type(name))
    #             st.write(name in car_df['Customer Name'] )
    #             st.write(car_df[name])
