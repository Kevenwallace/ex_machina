import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

SEED = 20

data = pd.read_csv(uri)

x = data[['home','how_it_works','contact']]
y = data['bought']


#treina na maior parte dos dados ex 75%
#testa na outra parte ex 25%
train_x , test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, random_state=SEED, stratify=y )

modelo = LinearSVC()
modelo.fit(train_x, train_y)
previsoes = modelo.predict(test_x)
resultado = accuracy_score(test_y, previsoes)
print(f"resultado foi de: {(resultado * 100):.2f}%")