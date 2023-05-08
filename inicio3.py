import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

url = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
data = pd.read_csv(url)


troca = {0:1, 1:0}

data['finished'] = data.unfinished.map(troca)


sns.scatterplot(x='expected_hours', y='price', data=data)

x = data[['expected_hours', 'price']]
y = data['finished']
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify=y)

print(f"treinamos com {len(train_x)}elementos e testamos com {len(test_x)}")

modelo = LinearSVC()
modelo.fit(train_x, train_y)
previsoes = modelo.predict(test_x)
resultado = accuracy_score(test_y, previsoes)
print(f"resultado foi de: {(resultado * 100):.2f}%")


previsoea_de_base = np.ones(len(test_x))
resultado = accuracy_score(test_y, previsoea_de_base)
print(f"resultado das previsoes de base foi de: {(resultado * 100):.2f}%")

#-curva_de_aprendizagem
sns.scatterplot(x='expected_hours', y='price',hue=test_y, data=test_x)

x_min = test_x.expected_hours.min()
x_max = test_x.expected_hours.max()

y_min = test_x.price.min()
y_max = test_x.price.max()

print(x_min, x_max, y_min, y_max)

pixel = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixel)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixel)

xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]
z = modelo.predict(pontos)
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(test_x.expected_hours, test_x.price, c=test_y, s=1)

