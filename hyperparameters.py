from xgboost import XGBClassifier # Clasificador de XGBoost
from bayes_opt import BayesianOptimization # Optimización Bayesiana
from sklearn.model_selection import cross_val_score, train_test_split # Cross Validation y partición del dataset en Train y Test
from sklearn.metrics import accuracy_score # Para medición del accuracy una vez que hagamos el XGBoost con los mejores parametros que nos da la optimización


# Limites inferiores y superiores de los hiperparametros que vamos a optimizar.
pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (100, 1000),
    'max_depth': (3,10),
    'subsample': (1.0, 1.0),  # Subsample lo dejamos en 1 que es default para XGBoost ya que no son muchos datos.
    'colsample': (1.0, 1.0),  # Como no hay muchas variables también lo dejamos en uno.
    'gamma': (0, 5)}

# División del dataset en train y test. Para la variable target vamos a usar como dijimos resultado2
# El dataset será dividido en 60 % para train y 40 % para test.
y = resultado2
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.4, random_state = 1)

# Función de optimización de los hiperparametros
def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample,
                        gamma):
    # Se transforman max_depth y n_estimators en int ya que XGBoost no acepta float.
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    
    # Instanciación del XGBClassifier con objetivo multi clasificación
    clf = XGBClassifier(objective="multi:softprob",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma)
    
    # Retornamos el valor de accuracy obtenido por el modelo
    return np.mean(cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy'))


# Instanciacion de la optimización bayesiana.
optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(init_points=20, n_iter=4)

print('Mejor Resultado:', optimizer.max)

modelo = XGBClassifier(
    objective="multi:softprob",
        max_depth=7,
        learning_rate=0.3041,
        n_estimators=294,
        gamma=0.4796)

modelo.fit(X_train, y_train)

# Predecimos los valores del conjunto de testeo y lo almacenamos en una variable para ver su accuracy
y_predict = modelo.predict(X_test)

print("Accuracy del modelo:", accuracy_score( y_test, y_predict))