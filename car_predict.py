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

from eda_app import run_eda_app
from ml_app import run_ml_app

def main():
    st.title('자동차 가격 예측')

    menu = ['Home', 'EDA', 'ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write('이 앱은 고객데이터와 자동차 구매 데이터에 대한 내용입니다.')
        st.write('고객의 정보를 입력하면, 어떤 가격의 차를 구매할 수 있는지 예측해주는 인공지능 앱입니다.')

        st.write('왼쪽의 사이드바에서 선택하세요')

    elif choice == 'EDA' :
        run_eda_app()

    elif choice == 'ML':
        run_ml_app()

if __name__ == '__main__':
    main()