import pandas as pd
import streamlit as st
import joblib

# Carga del modelo entrenado  (clf clasificador)
clf = joblib.load(filename='model/housing_model.pkl')

# Carga de los datos
df = pd.read_csv('data/housing.csv')
df = df.sample(frac=1).reset_index(drop=True)


style_width = """
    <style>
        .appview-container  .main  .block-container{
            max-width: 60%;
        }
    </style>
    """
st.markdown(style_width, unsafe_allow_html=True)

texto_para_ocultar = f"""
            longitude: Medida de la distancia al oeste de una casa; un valor más alto significa que está más al oeste.\n
            latitude: Medida de la distancia al norte de una casa; un valor más alto significa que está más al norte.\n
            housing_median_age (hmv): Edad media de una casa dentro de un bloque; un número más bajo corresponde a un edificio más nuevo.\n
            total_rooms (tr): Número total de habitaciones en un bloque.\n
            total_bedrooms (tb): Número total de dormitorios en un bloque.\n
            population: Número total de personas que viven en un bloque.\n
            households (hh): Número total de hogares, grupo de personas que residen en una unidad doméstica, de un bloque.\n
            median_income (mi): Renta media de los hogares de un bloque de viviendas (medida en decenas de miles de dólares estadounidenses).\n
            median_house_value (mhv): Valor medio de la vivienda de los hogares de una manzana (en dólares estadounidenses).\n
            ocean_proximity (ocpx): Situación de la casa con respecto al océano/mar.\n
        """


st.title('Find your future house!')

with st.expander("Descripción de los campos"):
    st.write(texto_para_ocultar)
col1, col2, col3 = st.columns(3)
with col1:
    longitude = st.number_input('Enter longitude:', key='longitude')
    latitude = st.number_input('Enter latitude:', key='latitude')
    housing_median_age = st.number_input('Enter housing median age:', key='housing_median_age')
    total_rooms = st.number_input('Enter total rooms:', key='total_rooms')
    total_bedrooms = st.number_input('Enter total bedrooms:', key='total_bedrooms')
with col2:
    population = st.number_input('Enter population:', key='population')
    households = st.number_input('Enter households:', key='households')
    median_income = st.number_input('Enter median income:', key='median_income')
    # median_house_value = st.number_input('Enter median house value:')
with col3:
    ocean_proximity = st.selectbox('Select ocean proximity', 
                                   ('<1H OCEAN', 
                                    'INLAND', 
                                    'NEAR OCEAN', 
                                    'NEAR BAY', 
                                    'ISLAND'), key='ocean_proximity')
# height = st.number_input('Enter height:')
# weight = st.number_input('Enter weight:')

# eye_color = st.selectbox( 'Select eye color', 
#                          ('Brown','Blue'))


if st.button('Submit'):
#     # Al enviar, el modelo va a realizar una predicción con el modelo y los datos que el usuario le proporciona.
    X = pd.DataFrame([[longitude,
                       latitude,
                       housing_median_age,
                       total_rooms,
                       total_bedrooms,
                       population,
                       households,
                       median_income,
                       ocean_proximity]], 
                       columns=[
                           'longitude',
                           'latitude',
                           'housing_median_age',
                           'total_rooms', 
                           'total_bedrooms', 
                           'population', 
                           'households', 
                           'median_income',
                           'ocean_proximity'])

    # X = X.replace(['Brown', 'Blue'], [1, 0])
    X = X.replace(['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], [0., 1., 2., 3., 4.])
    pred = clf.predict(X)[0]
    st.text(f'Your house value is ${pred}')
    # st.write('Your pet is a cat!')

st.write('Todos los datos sobre viviendas incluidas en el dataset')
st.write(df)
st.divider()

# col1, col2, col3 = st.columns(3)

# with col1:
#     color = st.color_picker('Selecciona un color', '#3475B3')

# with col2:
#     nombre = st.toggle('Mostrar el nombre')

# with col3:
#     sueldo = st.toggle('Mostrar el sueldo en la barra')

# with st.container():
#     if not df.empty:
#         sns.set_style("white")
#         plt.figure(figsize=(10, 8))
#         plot = sns.barplot(x="salary", y="full name", data=df, color=color)

#         if nombre:
#             # Mostrar el full name de la gráfica:
#             plt.yticks(df.index, df['full name'], fontsize=10)
#         else:
#             # Ocultar el full name de la gráfica:
#             plt.yticks([])

#         if sueldo:
#             for i in range(len(df)):
#                 plt.text(df.salary[i], i, df.salary[i], ha='left', va='center', color='black', fontsize=10)
#                 print(i, df.salary[i])

#         # Invierte todos los resultados del eje y:
#         plt.gca().invert_yaxis()

#         # sin etiquetas
#         plt.xlabel('')
#         plt.ylabel('')
#         plt.title("Salario de empleados")

#         plt.grid(axis='x', linestyle='-')

#         plt.xlim(0, 4500)
#         plt.show()
#         st.pyplot(plot.get_figure())
