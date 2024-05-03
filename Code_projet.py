# ******************************************* PARTIE 1 ********************************************************
#1/Chargements des données  

from ucimlrepo import fetch_ucirepo
 
# Récupérer l'ensemble de données
statlog_german_credit_data = fetch_ucirepo(id=144)
 
# Les données (sous forme de dataframes pandas)
x = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

#2/Description des données
print(statlog_german_credit_data.variables[['name','description']])

#Renommage
x.columns = ['statut_compte', 'duree_mois', 'historique', 'objectif',
             'montant', 'epargne', 'emploi', 'taux_versement', 'statut_sexe',
             'autre_debiteurs', 'residence', 'propriete', 'age', 'plan_versement',
             'logement', 'n_credit', 'travail', 'n_p_charge', 'tel', 'trav_etranger'] # Changer le nom des colonnes

# Résumé statistique variables
print(x.describe())  
print(x.info())

# Résumé statistique variable
y.columns = ['Infos']
print(y.describe())  
print(y.info())    


#3/Visualisation des données

# Importation des librairies à utiliser pour la visualisation des données
import matplotlib.pyplot as plt
import seaborn as sns
palette = "Set2"

#Univariée

# Pour une variable 'Montant' (quantitative)
plt.hist(x['montant'], bins=20)  # Création d'un histogramme 
plt.title('Répartition des montants')  # titre du graphique
plt.xlabel('Montant')  # axe des abscisses
plt.ylabel('Fréquence')  # axe des ordonnées
plt.show()  # Affichage du graphique

"""
La distribution des montants de crédit suggère que la majorité des prêts se situent dans la fourchette de 500€ à 4500€, avec une concentration particulière autour de ces deux valeurs. Un pic significatif est observé autour de 2000€, indiquant une fréquence plus élevée de crédits accordés à cette valeur spécifique.
"""

# Diagramme en barres pour la variable 'emploi'
sns.countplot(x='emploi', data=x, palette='viridis')
plt.title('Distribution des emplois')
plt.xlabel('Emploi')
plt.ylabel('Nombre d\'individus')
plt.show()

"""
'A73' a une distribution plus élevé.
"""

# Pour la variable 'objectif'(qualitative)
sns.countplot(x='objectif', data=x, order=x['objectif'].value_counts().index, palette=palette)  # Création d'un diagramme à barres 
plt.title('Répartition des objectifs')  # titre du graphique
plt.xlabel('Objectifs')  # axe des abscisses
plt.ylabel('Nombre')  # axe des ordonnées
plt.xticks(rotation=45)  # Rotation des étiquettes 
plt.show()  # Affichage du graphique

"""
'A43' a une fréquence significativement élevée dans l'échantillon. À l'inverse, 'A48' se distingue comme la catégorie d'objectifs la moins fréquente.
"""


#Bi-variée

sns.boxplot(x='logement', y='montant', data=x)  # Crée un box plot 
plt.title('Montant du crédit en fonction du logement') # titre du graphique
plt.xlabel('Logement') # axe des x 
plt.ylabel('Montant du Crédit')  # axe des y
plt.show()  # Affiche le graphique

"""
L'observation indique que les montants de crédit tendent à être plus élevés chez les individus bénéficiant d'un logement gratuit. En revanche, les médianes entre les locataires et les propriétaires semblent être relativement similaires. Cela suggère que la distinction principale en termes de montants de crédit se situe entre ceux qui ont un logement gratuit et les autres, tandis que la différence entre locataires et propriétaires est moins marquée en termes de médianes.
"""


# Matrice de corrélation entre les variables numériques
correlation_matrix = x[['montant', 'duree_mois', 'age', 'n_credit', 'taux_versement']].corr()

# Heatmap de la corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.show()


# Pairplot pour visualiser les relations entre plusieurs variables
sns.pairplot(x[['montant', 'duree_mois', 'age', 'n_credit', 'taux_versement']])
plt.show()



#4/Scinder en qualitatif et quantitatif
data_qualitative = x.select_dtypes(include=['object', 'category'])# Sélectionne les données qualitatives
data_quantitative = x.select_dtypes(include=['int64', 'float64'])# Sélectionne les données quantitatives

#5/ACP
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Standardiser les données quantitatives
data_quantitative_scaled = StandardScaler().fit_transform(data_quantitative)

# Appliquer l'ACP
pca = PCA()
principal_components = pca.fit_transform(data_quantitative_scaled)

# Créer un DataFrame avec les composantes principales pour la visualisation
df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

# Afficher la variance expliquée par chaque composante principale
print(pca.explained_variance_ratio_)

#6 Clustering


# Initialiser KMeans avec un `n_init` explicite pour éviter le FutureWarning
n_clusters = 3  # Exemple, ajustez selon vos besoins
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # Définition explicite de `n_init` et ajout de random_state
kmeans.fit(df)
labels = kmeans.labels_

# Ajouter les étiquettes de cluster au DataFrame des composantes principales
df['cluster'] = labels

# Visualiser les clusters sur le premier plan factoriel
plt.figure(figsize=(10, 6))
# Créer un scatter plot pour visualiser les clusters sur les deux premières composantes principales
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df, palette='Set1')
plt.title('Clustering k-means projeté sur le premier plan factoriel de l’ACP')
plt.show()


"""
Cette interprétation suggère que les points dans le graphique des composantes principales se concentrent principalement dans la partie inférieure, indiquant une forte contribution à la PC2.
"""

# ******************************************* PARTIE 2 ********************************************************
#Q1 : Expliquer les deux fonctions, randomUnitVector et svd_1d
def svd_1d(X, epsilon=1e-10):
    A = np.array(X)
    
    n, m = A.shape
    
    # Transformation de A en une  matrice stochastique de dimension (n+m) * (n+m)
    Dr = np.diag(A.sum(axis=1))
    Dc = np.diag(A.sum(axis=0))
    
    Dc_1 = np.linalg.inv(Dc)
    Dr_1 = np.linalg.inv(Dr)
    
    col1 = np.concatenate([np.zeros((n,n)), np.dot(Dc_1 , A.T)])
    col2 = np.concatenate([np.dot(Dr_1 , A), np.zeros((m,m))])
    
    S = np.concatenate([col1, col2], axis=1)
    
    # initialisation du vecteur currentV
    x = randomUnitVector(n+m)
    lastV = None
    currentV = x
    
    lastE = np.linalg.norm(currentV)

    # Itérations 
    iterations = 0
    while True:
        iterations += 1
        lastV = np.array(currentV)
        currentV = np.dot(S, lastV)
        currentV = currentV / norm(currentV)
        
        last_u = lastV[list(range(0,n))]
        last_v = lastV[list(range(n,n+m))]
        
        current_u = currentV[list(range(0,n))]
        current_v = currentV[list(range(n,n+m))]
        
        e_u = np.linalg.norm(current_u - last_u)
        e_v = np.linalg.norm(current_v - last_v)
        
        currentE = e_u + e_v
        
        d = abs(currentE - lastE)
        lastE = currentE
        
        if d <= epsilon:
            print("converged in {} iterations!".format(iterations))

            #u = currentV[range(0,n)]
            #v = currentV[range(n,n+m)]
            
            return current_u, current_v
"""
La fonction 'randomUnitVector' utilise des nombres aléatoires pour générer un vecteur aléatoire,
puis normalise ce vecteur pour qu'il devienne un vecteur unitaire.

La fonction 'svd_1d' semble implémenter une approximation de la décomposition en valeurs singulières (SVD)
pour une matrice bidimensionnelle donnée X.
"""

#Q2 : Créer une dataframe à partir de X en considérant le noms des lignes et colonnes?

X = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

columns=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
index=["HighSchool", "AgricultCoop", "Railstation", "OneRoomSchool", "Veterinary", "NoDoctor", "NoWaterSupply",  "PoliceStation", "LandReallocation"]

df = pd.DataFrame(X, index = index, columns = columns)


#Q3: Visualiser la matrice X (utiliser la fonction imshow de matplotlib)

plt.imshow(df, cmap= 'binary')


#Q4:Ecrire la fonction Reordonner (aide : utiliser argsort() de numpy)
   
def reordonner(X,u,v):
    u_trié = np.argsort(u)
    v_trié = np.argsort(v)
   
    X_reordonnee =X[np.ix_(u_trié,v_trié)]
   
    return X_reordonnee


#Q5:
    #Claculer les deux premiers vecteurs singuliers de X u et v - (appeler la fonction R1svd)
    #Réordonner X en fonction des tri de u et v- (appler la fonction Reordonner)
    #Visualiser X réordnnée (avec imshow) pour avoir une matrice réorganiser par blocs

u,v=svd_1d(X)
X_ré = reordonner (X,u,v)
plt.imshow(X_ré, cmap= 'binary')

#Q6:
    #Calculer la matrice de similarité entre les lignes de X,  𝑆𝑅=𝑋𝑋𝑇, ( 𝑋𝑇 est matrice transposée de X).
    #Calculer la matrice de similarité entre les colonnes de X,  𝑆𝐶=𝑋𝑇𝑋                        
    #Construire la matrice  𝑀=([𝑆𝑅  𝑋],[𝑋𝑇 𝑆𝐶])
    #Visualiser M reordonnée en fonction des tris de u et v.


SR = np.dot(X,np.transpose(X))
SC = np.dot(np.transpose(X),X)

M = np.concatenate([np.concatenate([SR, X], axis=1), np.concatenate([np.transpose(X), SC], axis=1)], axis=0)

u2,v2=svd_1d(M, epsilon=1e-10)
M_ré = reordonner (M,u2,v2)
plt.imshow(M_ré, cmap= 'binary')