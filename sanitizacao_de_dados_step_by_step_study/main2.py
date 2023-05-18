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


x_dados_v1 = data_base.drop(['diagnostico', 'exame_33'], axis= 1)
#exames era uma coluna predominantemente NaN
y_diagnostico = data_base['diagnostico']

# print(x.isnull().sum()) para verificacao


treino_x, teste_x, treino_y, teste_y = train_test_split(x_dados_v1,
                                                        y_diagnostico,
                                                        test_size=0.25
                                                    )

from sklearn.ensemble  import RandomForestClassifier

# classificados = RandomForestClassifier(n_estimators= 100)
classificados = RandomForestClassifier(n_estimators= 100)
classificados.fit(treino_x, treino_y)
print(f"resultado da classificacao: {(classificados.score(teste_x, teste_y)*100)}%")

from sklearn.dummy import DummyClassifier

np.random.seed(SEED)

classificadorDumy = DummyClassifier(strategy = "most_frequent")
classificadorDumy.fit(treino_x, treino_y)
print(f"resultado da classificacao dumy: {(classificadorDumy.score(teste_x, teste_y) * 100):.2f}%")




#Plot violino para observar features constantes:::
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

padronizador = StandardScaler()
padronizador.fit(x_dados_v1)
x_transformado = padronizador.transform(x_dados_v1)

x_transformado

#transforma novamente em data frame pois a padronização tira essa caracterisca
x_transformado1 = pd.DataFrame(data= x_transformado, columns=x_dados_v1.keys())


#execução do plot visualizavel no coolab ou jupter
dados_plot = pd.concat([ x_transformado1.iloc[:, 0:10],y], axis = 1)
dados_plot = pd.melt(dados_plot, id_vars = "diagnostico", var_name="exames", value_name="valores")
plt.figure(figsize=(10,10))
sns.violinplot(x = "exames", y = "valores",hue= "diagnostico", data = dados_plot, split=True)
plt.xticks(rotation = 90)

#funcao usada para limitar as features que serão plotadas
def check_violingraf(inicio, fim):
    padronizador = StandardScaler()
    padronizador.fit(x_dados_v1)
    x_transformado = padronizador.transform(x_dados_v1)

    x_transformado

    x_transformado1 = pd.DataFrame(data= x_transformado, columns=x_dados_v1.keys())

    dados_plot = pd.concat([ x_transformado1.iloc[:, inicio:fim],y], axis = 1)
    dados_plot = pd.melt(dados_plot, id_vars = "diagnostico", var_name="exames", value_name="valores")

    plt.figure(figsize=(10,10))
    sns.violinplot(x = "exames", y = "valores",hue= "diagnostico", data = dados_plot, split=True)

    plt.xticks(rotation = 90)



#apos a analise encontra-se  'exame_4','exame_29' duas features constantes
SEED = 1234

np.random.seed(SEED)

x_dados_v2 = data_base.drop(['diagnostico', 'exame_33', 'exame_4','exame_29' ], axis= 1)
#exames_33 era uma coluna predominantemente NaN 
y_diagnostico = data_base['diagnostico']

treino_x, teste_x, treino_y, teste_y = train_test_split(x_dados_v2,y_diagnostico, test_size=0.25)


# classificados = RandomForestClassifier(n_estimators= 100)
classificados = RandomForestClassifier(n_estimators = 100)
classificados.fit(treino_x, treino_y)
print(f"resultado da classificacao apos violinGrath: {(classificados.score(teste_x, teste_y)*100)}%")