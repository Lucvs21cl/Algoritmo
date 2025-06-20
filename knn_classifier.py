import kagglehub
teejmahal20_airline_passenger_satisfaction_path = kagglehub.dataset_download('teejmahal20/airline-passenger-satisfaction')

print('Importación de fuente de datos completada.')

"""# Clasificación de Satisfacción de Pasajeros de Aerolíneas usando K-Means

## Importación de Paquetes y Datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import joblib
from sklearn.metrics import silhouette_score

import os

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "test.csv")
data = pd.read_csv(csv_path, sep=",")

"""## Exploración de Datos"""

data.head()

"""**Exploración de data.info() y data.describe()**
- No hay datos nulos en ninguna columna
- Hay algunas características categóricas que necesitaremos codificar a números más adelante.
- Las columnas de Niveles de Satisfacción (Rango de 0 a 5) son: Servicio WiFi a bordo, Conveniencia de horarios de salida/llegada, Facilidad de reserva en línea, Ubicación de la puerta, Comida y bebida, Embarque en línea, Comodidad del asiento, Entretenimiento a bordo, Servicio a bordo, Servicio de espacio para piernas, Manejo de equipaje, Servicio de check-in, Servicio a bordo, y Limpieza
"""

data.info(verbose=True)

data.describe()

"""###

**Exploración de Características Categóricas**
"""

print(data['Gender'].value_counts(), '\n')
print(data['Customer Type'].value_counts(), '\n')
print(data['Type of Travel'].value_counts(), '\n')
print(data['Class'].value_counts(), '\n')
print(data['satisfaction'].value_counts(), '\n')

"""**Limpieza de Datos**
- Eliminar registros donde el servicio WiFi a bordo no era aplicable (0) ya que no había muchos. Mantener el 0 puede ser confuso ya que puede parecer que tiene una baja satisfacción.
- Eliminar registros donde la Clase era Eco Plus ya que no había muchos. Esto puede reducir algo de ruido para la segmentación del mercado.
- Codificar características categóricas a números.
- Eliminar columnas no utilizadas.
"""

print('Número de datos faltantes para servicio WiFi:', (data['Inflight wifi service'] == 0).sum())
print('Número de datos disponibles para servicio WiFi:', (data['Inflight wifi service'] != 0).sum())
data = data[data['Inflight wifi service'] != 0]
data = data[data['Class'] != 'Eco Plus']

data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
data['Customer Type'] = data['Customer Type'].map({'Loyal Customer': 0, 'disloyal Customer': 1})
data['Type of Travel'] = data['Type of Travel'].map({'Business travel': 0, 'Personal Travel': 1})
data['Class'] = data['Class'].map({'Business': 0, 'Eco': 1})
data['satisfaction'] = data['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

data = data.drop('id', axis=1)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

"""**Exploración de Correlaciones**"""

correlations = data.corr()
threshold = 0.5
high_correlations = correlations[(correlations >= threshold) | (correlations <= -1*threshold)]
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(high_correlations, annot=True, cmap='coolwarm')
plt.show()

"""**Transformación**
- Eliminar Retraso en Minutos de Llegada porque tiene una correlación de 0.97 con Retraso en Minutos de Salida.
- Escalar datos
- Reducir dimensiones
"""

data = data.drop(['Arrival Delay in Minutes'], axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=3, random_state=42)
pca.fit(data_scaled)
data_pca = pd.DataFrame(pca.transform(data_scaled), columns=(['col1', 'col2', 'col3']))
data_pca.describe().T

"""## Clustering"""

elbow = KElbowVisualizer(KMeans(n_init=10), k=10)
elbow.fit(data_pca)
elbow.show()

data_scaled = pd.DataFrame(data_scaled, columns = data.columns)

# Usar k = 4 del método del codo
model = KMeans(n_clusters=4, random_state=42, n_init=10)
predictions = model.fit_predict(data_pca)
data_pca['Cluster'] = predictions
data['Cluster'] = predictions
data_scaled['Cluster'] = predictions

#Graficar los clusters
x = data_pca['col1']
y = data_pca['col2']
z = data_pca['col3']
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label='bla')
ax.scatter(x, y, z, s=40, c=data_pca['Cluster'], marker='o', cmap = cmap )
plt.show()

"""## Evaluación"""

#Graficar conteo de clusters
print('Distribución relativamente equilibrada de clusters.')
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x = 'Cluster', data = data, hue = 'Cluster', palette = pal)
pl.set_title("Distribución de Clusters")
plt.show()

cluster = data.groupby(['Cluster']).mean().T
cluster.style.background_gradient(cmap='RdYlGn',axis=1)

# Índice de silueta
score = silhouette_score(data_pca, predictions)
print(f"Índice de silueta: {score:.3f}")

print(data['Cluster'].value_counts())

"""## Perfiles

#### Grupo 0 - Viajeros de Negocios Frecuentes/Leales
- Demografía:
    - Edad: Personas mayores
    - Tipo de Cliente: Mayoría de Clientes Leales
    - Tipo de Viaje: Mayoría de Viajes de Negocios
    - Clase: Mayoría de Clases de Negocios
    - Distancia de Vuelo: Las distancias más largas
- Este grupo es el menos crítico/más apreciativo/más indulgente en todas las categorías de satisfacción
- Otros:
    - Menor retraso en minutos
    - Satisfacción más alta
- Notas: Los viajeros de negocios frecuentes/leales contribuyen con la menor cantidad de minutos a los retrasos de vuelo. Sus muchas experiencias de vuelo pasadas pueden contribuir a una mayor indulgencia y
comprensión de la experiencia general de vuelo, lo que puede contribuir a los niveles más altos de satisfacción general.

#### Grupo 1 - El Grupo Joven
- Demografía:
    - Grupo de edad: El más joven
    - Tipo de Cliente: Mayoría de Clientes Desleales
    - Tipo de Viaje: Mayoría de Viajes Personales
    - Clase: Mayoría de Clases Económicas
    - Distancia de Vuelo: Las distancias más cortas
- Este grupo es el más crítico con:
    - Comida y bebida
    - Embarque en línea
    - Comodidad del asiento
    - Entretenimiento a bordo
    - Limpieza
- Otros:
    - Mayor retraso en minutos
    - Satisfacción más baja
- Notas: Los jóvenes en vacaciones con un presupuesto más pequeño tienden a preocuparse más por su experiencia cinematográfica durante el vuelo.
Parece que también tienden a causar retrasos y son los más difíciles de complacer.

#### Grupo 2 - Viajeros de Negocios Poco Frecuentes
- Demografía:
    - Grupo de edad: Más personas mayores
    - Tipo de Viaje: Más hacia Viajes de Negocios
    - Clase: Más hacia Clases de Negocios
    - Distancia de Vuelo: Más hacia distancias largas
- Este grupo es el más crítico con:
    - Servicio WiFi a bordo
    - Conveniencia de horarios de salida/llegada
    - Facilidad de reserva en línea
    - Ubicación de la puerta
- Este grupo es apreciativo con:
    - Comida y bebida
    - Embarque en línea
    - Comodidad del asiento
    - Entretenimiento a bordo
    - Servicio a bordo
    - Servicio de espacio para piernas
    - Manejo de equipaje
    - Servicio de check-in
    - Servicio a bordo
    - Limpieza
- Notas: Los viajeros de negocios poco frecuentes que viajan quizás una o dos veces al año. Su falta de experiencia de vuelo se refleja en su crítica a la programación/logística del vuelo y a la facilidad tecnológica para viajar.
Como están en un viaje de negocios profesional, pueden estar más inclinados a ser más apreciativos de los trabajadores del servicio durante la experiencia de vuelo.

#### Grupo 3 - Viajeros Familiares Poco Frecuentes
- Demografía:
    - Grupo de edad: Más personas mayores
    - Tipo de Viaje: Más Viajes Personales
    - Clase: Más Clases Económicas
- Este grupo es el más crítico con:
    - Servicio a bordo
    - Servicio de espacio para piernas
    - Manejo de equipaje
    - Servicio de check-in
    - Servicio a bordo
- Notas: Padres económicos en vacaciones que ya pueden estar estresados por cuidar a sus hijos y pueden estar buscando un buen servicio y alojamientos
para disminuir su nivel de estrés. Una vez que están a bordo, no son tan críticos excepto por el espacio para las piernas, lo que podría ser que están llevando cosas extra
o necesitan espacio adicional para cuidar a sus hijos. Tienen otras cosas de qué preocuparse además de la comida, la limpieza y el entretenimiento a bordo.
"""

# Guardar el modelo y el scaler
import joblib

# Guardar el modelo KMeans
joblib.dump(model, os.path.join(current_dir, 'kmeans_model.joblib'))
# Guardar el scaler
joblib.dump(scaler, os.path.join(current_dir, 'scaler.joblib'))
# Guardar el PCA
joblib.dump(pca, os.path.join(current_dir, 'pca.joblib'))
# Guardar el orden de las columnas (excluyendo 'Cluster')
columnas_para_guardar = [col for col in data.columns if col != 'Cluster']
joblib.dump(columnas_para_guardar, os.path.join(current_dir, 'columnas.joblib'))

print("Modelo, scaler, PCA y orden de columnas guardados exitosamente.")

print(data.shape)