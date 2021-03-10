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
    new_data = np.array([0, 38, 90000, 2000, 500000])
    new_data = new_data.reshape(1,-1)

    sc_X = joblib.load('data/sc_X.pkl')
    new_data = sc_X.transform(new_data)

    y_pred = model.predict(new_data)

    sc_y = joblib.load('data/sc_y.pkl')
    y_pred = sc_y.inverse_transform(y_pred)

    st.write(y_pred)

