import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/alura-cursos/reducao-dimensionalidade/master/data-set/exames.csv"
data_base = pd.read_csv(url)
data_base.head()

x = data_base.drop(['diagnostico'], axis= 1)
y = data_base['diagnostico']


# print(x.shape) para verificacao
# print(x.isnull().sum()) para verificacao

print(f"Shape quantidade de colunas é 569, e quantidade de nulls na coluna 'exame_33' é de 419")
print(f"ou seja {(419 / 569) *100:.2f}%")

# SEED = 123 # seed de "123" ficou mt facil
SEED = 1234

np.random.seed(SEED)


x = data_base.drop(['diagnostico', 'exame_33'], axis= 1)
#exames era uma coluna predominantemente NaN
y = data_base['diagnostico']

# print(x.isnull().sum()) para verificacao


treino_x, teste_x, treino_y, teste_y = train_test_split(x,
                                                        y,
                                                        test_size=0.25
                                                    )

from sklearn.ensemble  import RandomForestClassifier

# classificados = RandomForestClassifier(n_estimators= 100)
classificados = RandomForestClassifier(n_estimators= 100)
classificados.fit(treino_x, treino_y)
print(f"resultado da classificacao: {(classificados.score(teste_x, teste_y)*100):.2f}%")

from sklearn.dummy import DummyClassifier

np.random.seed(SEED)

classificadorDumy = DummyClassifier(strategy = "most_frequent")
classificadorDumy.fit(treino_x, treino_y)
print(f"resultado da classificacao dumy: {(classificadorDumy.score(teste_x, teste_y) * 100):.2f}%")