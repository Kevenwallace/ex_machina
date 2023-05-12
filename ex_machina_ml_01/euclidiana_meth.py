import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
data_final.head()

xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

data_final.shape
pd.set_option('display.max_columns',39)
data_final.head()


x = data_final.drop('Churn', axis = 1)
y = data_final['Churn']


norm  = StandardScaler()

x_normalizado = norm.fit_transform(x)
x_normalizado[0]

xmaria_normalizado = norm.transform(pd.DataFrame(xmaria, columns = x.columns))
xmaria_normalizado

#Distancia euclidiana calc
a = xmaria_normalizado
b = x_normalizado[0]

#-1-passo- subtração
a - b
#-2-passo- exponenciação
np.square(a-b)
#-3-passo somatorio
somatorio = np.sum(np.square(a-b))

x_treino, x_teste, y_treino, y_teste = train_test_split(
                                        x_normalizado, y,
                                        test_size=0.3,
                                        random_state=123
                                        )


knn = KNeighborsClassifier(metric='euclidean')
knn.fit(x_treino, y_treino)

predito = knn.predict(x_teste)
print(predito)
data = accuracy_score(y_teste, predito)
print(f"previsao de {data * 100:.2f}%")