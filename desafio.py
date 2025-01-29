#!/usr/bin/env python
# coding: utf-8

# # Previsão de Preços de Aluguéis Temporários em Nova York
# 
# ## Objetivo
# Este notebook tem como objetivo desenvolver um modelo preditivo para estimar preços de aluguéis temporários em Nova York com base em dados históricos. O processo inclui:
# 1. Análise exploratória de dados (EDA) para entender as variáveis e seus impactos.
# 2. Pré-processamento para limpar e transformar os dados.
# 3. Modelagem preditiva, com testes e avaliação de diferentes algoritmos.
# 4. Geração de previsões e salvamento do modelo final.

# In[6]:


# Importação de Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import pickle
from IPython.display import display

# Configurações Gerais
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)


# In[ ]:


# Carregando os Dados
data_path = "data/teste_indicium_precificacao.csv"  # Substitua pelo caminho correto do arquivo
df = pd.read_csv(data_path)

# Inspeção Inicial
print("Dimensões do dataset:", df.shape)
display(df.head())
print("\nInformações gerais:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())


# In[ ]:


# Distribuição de Preços
sns.histplot(df['price'], bins=50, kde=True, color='blue', edgecolor='black')
plt.title("Distribuição de Preços dos Aluguéis", fontsize=16)
plt.xlabel("Preço (USD)", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.show()

# Preços por Tipo de Quarto
sns.boxplot(data=df, x="room_type", y="price", palette="coolwarm")
plt.title("Distribuição de Preços por Tipo de Quarto", fontsize=16)
plt.xlabel("Tipo de Quarto", fontsize=12)
plt.ylabel("Preço (USD)", fontsize=12)
plt.show()

# Correlação entre Variáveis
numeric_df = df.select_dtypes(include=[np.number])
correlacoes = numeric_df.corr()
sns.heatmap(correlacoes, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()


# In[ ]:


# Criando variáveis derivadas
def extrair_palavras_chave(nome):
    palavras_chave = ['luxury', 'exclusive', 'cozy', 'modern', 'spacious']
    nome = nome.lower() if isinstance(nome, str) else ""
    return any(palavra in nome for palavra in palavras_chave)

df['possui_palavras_chave'] = df['nome'].apply(extrair_palavras_chave).astype(int)

# Removendo outliers
limite_inferior = df['price'].quantile(0.01)
limite_superior = df['price'].quantile(0.99)
df = df[(df['price'] >= limite_inferior) & (df['price'] <= limite_superior)]

# Separando dados
X = df.drop(['price', 'id', 'nome', 'host_name', 'ultima_review'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurando as variáveis categóricas e numéricas
cat_features = ['bairro_group', 'room_type']
num_features = ['latitude', 'longitude', 'minimo_noites', 'numero_de_reviews', 
                'reviews_por_mes', 'calculado_host_listings_count', 'disponibilidade_365']

# Atualizando o pipeline para lidar com valores ausentes nas variáveis numéricas
cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Preenchendo valores ausentes com a mediana
    ('scaler', StandardScaler())                   # Padronizando os valores
])

# Combinando as transformações
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features), 
    ('cat', cat_pipeline, cat_features)
])

# Aplicando o pipeline de pré-processamento
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print("Pré-processamento concluído com sucesso!")


# In[ ]:


import nbformat as nbf

# Crie um novo notebook
nb = nbf.v4.new_notebook()

# Adicione células ao notebook
nb.cells = [
    nbf.v4.new_markdown_cell("# Desafio"),
    nbf.v4.new_code_cell("print('Olá, mundo!')"),
    nbf.v4.new_markdown_cell("## Instruções"),
    nbf.v4.new_markdown_cell("Resolva os problemas abaixo:"),
    nbf.v4.new_code_cell("# Adicione seu código aqui\n")
]

# Salve o notebook no arquivo 'desafio.ipynb'
with open("desafio.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Arquivo 'desafio.ipynb' salvo com sucesso!")


# In[ ]:


# Definindo a função de avaliação do modelo
def avaliar_modelo(model, X_train, y_train, X_test, y_test):
    # Avaliação no conjunto de treinamento
    y_train_pred = model.predict(X_train)
    print("Conjunto de Treinamento:")
    print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
    print(f"R2: {r2_score(y_train, y_train_pred):.2f}")
    
    # Avaliação no conjunto de teste
    y_test_pred = model.predict(X_test)
    print("\nConjunto de Teste:")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_test_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
    print(f"R2: {r2_score(y_test, y_test_pred):.2f}")

# Treinando o modelo XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_preprocessed, y_train)

# Avaliando o modelo
print("\n### Avaliação do Modelo: XGBoost ###")
avaliar_modelo(xgb_model, X_train_preprocessed, y_train, X_test_preprocessed, y_test)

# Previsão para o exemplo fornecido
exemplo = pd.DataFrame({
    'bairro_group': ['Manhattan'],
    'room_type': ['Entire home/apt'],
    'latitude': [40.75362],
    'longitude': [-73.98377],
    'minimo_noites': [1],
    'numero_de_reviews': [45],
    'reviews_por_mes': [0.38],
    'calculado_host_listings_count': [2],
    'disponibilidade_365': [355],
    'possui_palavras_chave': [1]
})

# Pré-processando o exemplo
exemplo_preprocessed = preprocessor.transform(exemplo)

# Fazendo a previsão
preco_predito = xgb_model.predict(exemplo_preprocessed)
print(f"Preço previsto para o exemplo: ${preco_predito[0]:.2f}")


# In[ ]:


# Salvando o modelo final (XGBoost)
model_path = "models/final_model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(xgb_model, file)
print(f"Modelo salvo em: {model_path}")

