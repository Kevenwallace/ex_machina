import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

url = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

data = pd.read_csv(url)
troca = {0:1, 1:0}

data['finished'] = data.unfinished.map(troca)


x = data[['expected_hours', 'price']]
y = data['finished']
SEED = 20
np.random.seed(SEED)
raw_train_x, raw_test_x, raw_train_y, raw_test_y = train_test_split(x, y,  test_size=0.25, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

modelo = SVC(gamma='auto')
modelo.fit(train_x, raw_train_y)
previsoes = modelo.predict(test_x)
resultado = accuracy_score(raw_test_y, previsoes)
print(f"resultado foi de: {(resultado * 100):.2f}%")

dados_x = test_x[:,0]
dados_y = test_x[:,1]

x_min = dados_x.min()
x_max = dados_x.max()

y_min = dados_y.min()
y_max = dados_y.max()


pixel = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixel)

eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixel)


xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]
z = modelo.predict(pontos)
z = z.reshape(xx.shape)


plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(dados_x, dados_y, c=raw_test_y, s=1)