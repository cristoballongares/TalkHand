import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Cargar los datos de los archivos pickle
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Convertir los datos a un arreglo de numpy
X = np.array(features)
y = np.array(labels)

# Crear un pipeline de clasificación con PCA y SVM
clf = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC(kernel='linear', C=1, gamma='auto'))
])

# Entrenar el modelo de clasificación múltiples veces
n_iterations = 100
accuracies = []
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f'Iteración {i+1}: Precisión del modelo: {acc}')

# Imprimir la precisión promedio y la desviación estándar del modelo
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f'Precisión promedio del modelo: {mean_acc}')
print(f'Desviación estándar de la precisión: {std_acc}')

# Guardar el modelo entrenado
with open('my_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


