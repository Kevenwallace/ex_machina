import pandas as pd
import seaborn as sns
import numpy as np
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