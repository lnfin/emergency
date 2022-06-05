import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import torch
from utils import get_data
from catboost import CatBoostRegressor, Pool
from PIL import Image


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


def main():
    st.title('Оптимизация работы скорой помощи')

    # st.markdown("""
    # После загрузки данных не закрывайте страницу. Предсказание займет какое-то время.
    # """)
    st.subheader('Загрузка файлов')

    uploaded_file = st.file_uploader("Выберите файл")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        lat_lon_subs = pd.read_csv('substations_lat_lon.csv')

        st.subheader('Результат')
        with st.expander('Предсказания'):
            # st.subheader('Предсказания')

            with st.spinner('Данные обрабатываются, пожалуйста, подождите'):
                test_df, target = get_data()

            model = CatBoostRegressor(cat_features=['Подстанция', 'Сезон',
                                                    'quarter', 'dayofweek'], task_type='gpu')
            model = model.load_model("model_main.cbm")

            test_data = Pool(test_df,
                        cat_features=['Подстанция', 'Сезон',
                                      'quarter', 'dayofweek']
                        )

            with st.spinner('Делаются предсказания'):
                target['Предсказание'] = model.predict(test_data)

            sub_station = st.selectbox('Подстанция для визуализации', target['Подстанция'].unique())
            target_sum_day = target[target['Подстанция'] == sub_station].groupby('Дата').agg('sum').reset_index()

            fig = px.line(target_sum_day[(target_sum_day['Дата'] > '2022-04-10') & (target_sum_day['Дата'] < '2022-05-10')],
                          x=target_sum_day[(target_sum_day['Дата'] > '2022-04-10') & (target_sum_day['Дата'] < '2022-05-10')].index,
                          y=['Предсказание', "Загруженность"], title='Предсказания на 24 апреля - 24 мая', height=600,
                          labels=dict(x='Валидационный месяц'))
            st.plotly_chart(fig, use_container_width=True)

            fig = px.line(target_sum_day[(target_sum_day['Дата'] > '2022-05-10')],
                          x=target_sum_day[(target_sum_day['Дата'] > '2022-05-10')].index,
                          y=['Предсказание'], title='Предсказания на 24 мая - 7 июня', height=600,
                          labels=dict(x='Тестовый месяц'))
            st.plotly_chart(fig, use_container_width=True)

            csv = convert_df(target_sum_day[['Дата', 'Предсказание']])
            st.download_button('Скачать предсказания', csv, file_name='preds.csv')

        with st.expander('Полезные признаки'):
            # st.subheader('Полезные признаки')
            feature_importance = model.get_feature_importance(prettified=True)[:10]
            feature_importance.loc[4, 'Feature Id'] = 'Квартал'
            feature_importance.loc[5, 'Feature Id'] = 'Месяц'
            feature_importance.loc[7, 'Feature Id'] = 'Вызывавший человек'
            feature_importance.loc[8, 'Feature Id'] = 'Праздник в этот день'
            feature_importance.loc[9, 'Feature Id'] = 'Температура воздуха'
            feature_importance['Importances_log'] = np.log(feature_importance['Importances'])

            st.write('##### 10 признаков с наибольшим вкладом')
            fig = px.bar(feature_importance.rename(columns={'Feature Id': 'Название признака',
                                                            'Importances_log': 'Важность'}), x='Название признака', y='Важность', height=600)
            st.plotly_chart(fig, use_container_width=True)

            image = Image.open('age.jpg')
            st.write('##### Зависимость времени в пути от возраста пациента')
            st.image(image, width=650)

            image = Image.open('quater.jpg')
            st.write('##### Зависимость загруженности от квартала года')
            st.image(image, width=650)

            image = Image.open('dayofweek.jpg')
            st.write('##### Зависимость загруженности от дня недели')
            st.image(image, width=650)

        with st.expander('Анализ и статистика'):
            # st.subheader('Анализ и статистика')

            df = data.copy()
            df = df[~(df['Приезд'] == '-')]
            df['Приезд'] = df['Приезд'].fillna('00:00')
            df['Принят'] = df['Принят'].fillna('00:00')
            df = df.reset_index(drop=True)

            df['Приезд_minute'] = df['Приезд'].apply(lambda x: int(x[-2:]))
            df['Принят_minute'] = df['Принят'].apply(lambda x: int(x[-2:]))
            df['time_to'] = df['Приезд_minute'] - df['Принят_minute']
            df = df[~(df['time_to'] <= 0)]

            df['Дата'] = pd.to_datetime(df['Дата'])
            df['month'] = df['Дата'].dt.month
            df['year'] = df['Дата'].dt.year
            df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-01'

            df_substations = df.groupby(['Подстанция', 'date']).agg({'Приезд': 'count'}).reset_index()
            df_substations = df_substations[1:]
            df_substations = df_substations.merge(lat_lon_subs[['lat', 'lon', 'title']], left_on='Подстанция', right_on='title',
                                                  how='left')
            df_substations = df_substations[1:]

            df_substations['date'] = pd.to_datetime(df_substations['date'])
            df_substations = df_substations.sort_values('date')
            df_substations['date'] = df_substations['date'].astype(str)

            st.write('##### Тепловая карта по подстанциям за каждый месяц')

            fig = px.density_mapbox(lat=df_substations.lat, lon=df_substations.lon,
                                    z=df_substations['Приезд'], radius=25, opacity=1, zoom=0, height=600,
                                    animation_frame=df_substations["date"], animation_group=df_substations['title'],
                                    title='Тепловая карта подстанций по количеству времени за каждый месяц')

            fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=100)
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

            ############

            df_diag = df.groupby(['Диагноз', 'date']).agg({'Приезд': 'count'}).reset_index()
            df_diag = df_diag[~(df_diag['Диагноз'] == '-')]

            top_diag = (df.groupby(['Диагноз'])['Приезд'].count()).reset_index()
            top_diag = top_diag[~(top_diag['Диагноз'] == '-')]

            df_diag = df_diag[df_diag['Диагноз'].isin(top_diag.sort_values('Приезд')[-10:]['Диагноз'].values)]

            df_diag['date'] = pd.to_datetime(df_diag['date'])
            df_diag = df_diag.sort_values('date')
            df_diag['date'] = df_diag['date'].astype(str)
            df_diag = df_diag.sort_values('Приезд')

            st.write('##### 10 самых частных причин вызова скорой')

            fig = px.bar(df_diag.rename(columns={'Приезд': 'Кол-во вызовов'}), x="Диагноз", y="Кол-во вызовов", color="Кол-во вызовов",
                         animation_frame="date", animation_group="Диагноз", height=900, range_x=[-2, 10],
                         range_y=[0, 1700])
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
            fig['layout']['updatemenus'][0]['pad'] = dict(t=250)
            fig['layout']['sliders'][0]['pad'] = dict(t=250, )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()