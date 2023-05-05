from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
 #features
 #pelo longo
 #perna curta?
 #faz auau?

porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]


cachorro1 = [0,1,1]
cachorro2 = [1,0,1]
cachorro3 = [1,1,1]

# 1 = porco; 0 = cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1,1,1,0,0,0]

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes_x = [misterio1, misterio2, misterio3]
testes_y = [0, 1, 1]

previsoes = modelo.predict(testes_x)


taxa_de_acerto = accuracy_score(previsoes, testes_y)

print(f"taxa de resultado é igual a: {(taxa_de_acerto * 100):.2f}")