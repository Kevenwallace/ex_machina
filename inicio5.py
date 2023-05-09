import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz



troca = {
    'yes':1,
    'no':0
}


url = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
data = pd.read_csv(url)

data.sold = data.sold.map(troca)

from datetime import datetime
ano_atual = datetime.today().year
data['car_age'] = ano_atual - data.model_year

data = data.drop(columns= ['model_year'], axis=1)

x = data[['mileage_per_year', 'price', 'car_age']]
y = data['sold']

# SEED = 20
# np.random.seed(SEED)

# raw_train_x, raw_test_x, raw_train_y, raw_test_y = train_test_split(x, y, test_size=0.25, stratify=y)


# scaler = StandardScaler()
# scaler.fit(raw_train_x)
# train_x = scaler.transform(raw_train_x)
# test_x = scaler.transform(raw_test_x)


# modelo = SVC(gamma='auto')
# treino = modelo.fit(train_x, raw_train_y)
# predict = modelo.predict(test_x)
# resultado = accuracy_score(raw_test_y, predict)
# resultado

# dados_x = test_x[:,0]
# dados_y = test_x[:,1]

# x_min = dados_x.min()
# x_max = dados_x.max()

# y_min = dados_y.min()
# y_max = dados_y.max()


# pixel = 100

# eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixel)

# eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixel)


# xx, yy = np.meshgrid(eixo_x, eixo_y)

# pontos = np.c_[xx.ravel(), yy.ravel()]
# z = modelo.predict(pontos)
# z = z.reshape(xx.shape)


# plt.contourf(xx, yy, z, alpha=0.3)
# plt.scatter(dados_x, dados_y, c=raw_test_y, s=1)

# tese_de_base = np.ones(len(raw_test_x))


# from sklearn.dummy import DummyClassifier

# dummy = DummyClassifier()
# treino = dummy.fit(raw_train_x, raw_train_y)
# predict = dummy.predict(raw_test_x)
# resultado = accuracy_score(raw_test_y, predict)
# resultado



x = data[['mileage_per_year', 'price', 'car_age']]
y = data['sold']

SEED = 20
np.random.seed(SEED)

raw_train_x, raw_test_x, raw_train_y, raw_test_y = train_test_split(x, y, test_size=0.25, stratify=y)



modelo = DecisionTreeClassifier(max_depth=5)
treino = modelo.fit(raw_train_x, raw_train_y)
predict = modelo.predict(raw_test_x)
resultado = accuracy_score(raw_test_y, predict)
resultado
print("resultado foi de: ",resultado * 100, "%")
features = x.columns
dot_data = export_graphviz(modelo,
                           filled=True, rounded=True, class_names=['não', 'sim'],
                           out_file=None, feature_names = features)
grafico = graphviz.Source(dot_data)
grafico