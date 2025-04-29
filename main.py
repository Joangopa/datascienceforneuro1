import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
@st.cache_data
def load_data():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    # Filtrar os dados onde MMSE e CDR não são nulos
    data = data.dropna(subset=['MMSE', 'CDR'])
    return data

data = load_data()

# Título do dashboard
st.title('Dashboard de Análise de Dados OASIS')

# Exibir os dados brutos
if st.checkbox('Mostrar dados brutos'):
    st.write(data)

# Filtros interativos
st.sidebar.header('Filtros')
gender_filter = st.sidebar.selectbox('Selecione o gênero', ['Todos', 'Masculino', 'Feminino'])
age_filter = st.sidebar.slider('Selecione a faixa etária', int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())))

# Aplicar filtros
filtered_data = data
if gender_filter != 'Todos':
    filtered_data = filtered_data[filtered_data['M/F'] == gender_filter[0]]
filtered_data = filtered_data[(filtered_data['Age'] >= age_filter[0]) & (filtered_data['Age'] <= age_filter[1])]

# Exibir dados filtrados
st.write(f'Dados filtrados: {len(filtered_data)} registros')
st.write(filtered_data)

# Gráfico de dispersão: Idade vs MMSE
st.subheader('Gráfico de Dispersão: Idade vs MMSE')
fig, ax = plt.subplots()
sns.scatterplot(x='Age', y='MMSE', data=filtered_data, hue='M/F', ax=ax)
st.pyplot(fig)

# Histograma de Idade
st.subheader('Histograma de Idade')
fig, ax = plt.subplots()
sns.histplot(filtered_data['Age'], kde=True, ax=ax)
st.pyplot(fig)

# Gráfico de barras: CDR por Gênero
st.subheader('Gráfico de Barras: CDR por Gênero')
fig, ax = plt.subplots()
sns.countplot(x='CDR', hue='M/F', data=filtered_data, ax=ax)
st.pyplot(fig)

# Estatísticas descritivas
st.subheader('Estatísticas Descritivas')
st.write(filtered_data.describe())