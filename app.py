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
from keras.models import Sequential
from keras.layers import Dense

CHECKPOINT_PATH = 'C:/Users/5-14/Documents/Streamlit/day04/checkpoints/'+'.h5'


def main():
    st.title('고객 데이터를 이용해 자동차 추천하기')

    car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1' )
    st.subheader('전체 데이터셋')
    st.dataframe(car_df)

    st.subheader('데이터의 상관관계')
    st.dataframe(car_df.corr())
    
    st.subheader('상관관계 시각화')
    # sns.pairplot(data=car_df, vars=[ 'Age','Annual Salary', 'Credit Card Debt', 'Net Worth','Car Purchase Amount'])
    # plt.show()
    
    st.subheader('Nan값이 있는지 확인')
    st.write(car_df.isna().sum())

    X = car_df.loc[:, 'Gender':'Net Worth']
    st.subheader('사용할 데이터 추출')
    st.dataframe(X)

    y = car_df['Car Purchase Amount']
    st.subheader('결과값(자동차 가격) 추출')
    st.dataframe(y)

    enc = LabelEncoder()
    X['Gender'] = enc.fit_transform(X['Gender'])

    sc_X = MinMaxScaler()
    X_scaled = sc_X.fit_transform(X)

    st.subheader('머신러닝을 위한 정규화')
    st.dataframe(X_scaled)

    y = y.values

    y = y.reshape(-1,1)
    # 스케일링을 하려면 2차원이여야함.

    sc_y = MinMaxScaler()
    y_scaled = sc_y.fit_transform(y)

    st.subheader('학습을 위해 결과값도 정규화')
    st.dataframe(y_scaled)

    my_model = tf.keras.models.load_model(CHECKPOINT_PATH)

    gender = st.selectbox('성별을 선택하시오', ['남자','여자'])
    if gender =='남자':
        gen_num = 0
    if gender =='여자':
        gen_num = 1
    age = st.number_input('나이를 입력하세요')
    salary = st.number_input('연봉을 입력하세요(달러)')
    debt = st.number_input('카드빚을 입력하세요(달러)')
    worth = st.number_input('순 자산을 입력하세요(달러)')

    input_list = np.array([gen_num, age, salary, debt, worth])

    if st.button('입력한 값을 보여줍니다'):
        st.subheader('입력한 값을 보여줍니다')
        st.write(input_list)
        st.write(type(input_list))

    if st.button('고객이 살 차량의 금액을 원한다면 버튼을 눌러주세요'):
        input_list_reshape = input_list.reshape(1,-1)
        sc_input = sc_X.transform(input_list_reshape)
        y_pred = my_model.predict(sc_input)
        y_pred_origin = sc_y.inverse_transform(y_pred)
        st.write(int(y_pred_origin))

if __name__ == '__main__':
    main()
