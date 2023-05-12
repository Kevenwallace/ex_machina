import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB


url = "https://raw.githubusercontent.com/alura-cursos/ML_Classificacao_por_tras_dos_panos/main/Dados/Customer-Churn.csv"

data = pd.read_csv(url)

data.head()

change = {'Sim': 1,
          'Nao': 0}

dados_modificados = data[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline','Churn']].replace(change)
dados_modificados.head()

dumie_dados = pd.get_dummies(data.drop(
    ['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline','Churn'],
    axis=1))

data_final = pd.concat([dados_modificados, dumie_dados], axis=1)
print(data_final.head())

data_final.shape
pd.set_option('display.max_columns',39)
print(data_final.head())


x = data_final.drop('Churn', axis = 1)
y = data_final['Churn']


norm  = StandardScaler()

x_normalizado = norm.fit_transform(x)
print(x_normalizado[0])


x_treino, x_teste, y_treino, y_teste = train_test_split(
                                        x_normalizado, y,
                                        test_size=0.3,
                                        random_state=123
                                        )

media = np.median(x_treino * -1)
print(media) #tem que se positivo

bnb = BernoulliNB(binarize=media)

bnb.fit(x_treino, y_treino)
predito_bnb = bnb.predict(x_teste)
predito_bnb
data = accuracy_score(y_teste, predito_bnb)
print(f"previsao de {data * 100:.2f}%")