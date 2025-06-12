import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Charger les données
iris = load_iris()
X = iris.data  # Caractéristiques
y = iris.target  # Classes : 0, 1 ou 2

# 2. Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardiser les données (optionnel mais recommandé)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entraîner le modèle de régression logistique multiclasse
# multi_class='multinomial' => Softmax
# solver='lbfgs' => bon choix pour petits jeux de données
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_test_scaled)

print("Classification report :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 6. Afficher les probabilités (optionnel)
probas = model.predict_proba(X_test_scaled)
print("Probabilités :")
print(probas[:5])  # Affiche les 5 premières lignes
