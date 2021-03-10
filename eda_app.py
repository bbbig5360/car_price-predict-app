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

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1' )
    

    radio_menu = ['데이터 프레임', '통계치']
    selected_radio = st.radio('선택하세요', radio_menu)

    if selected_radio == '데이터 프레임':
        st.dataframe(car_df)

    if selected_radio == '통계치':
        st.dataframe(car_df.describe())
        


