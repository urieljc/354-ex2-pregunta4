# -*- coding: utf-8 -*-
"""
Created on Sun May 30 18:46:46 2021

@author: BazanJuanCarlos
"""

#lectura del dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset=pd.read_csv("audit_risk.csv")
#dataset=pd.read_csv("trial.csv")
print( dataset.head())
print("______________________________________________________")
print("mostrando las columnas del dataset ")
print(dataset.columns)
print("______________________________________________________")
print("mostrando las caracteristicas de cada columna")
print( dataset.dtypes)
#preprocesamiento
print("_______________________________________________________")
print("PREPROCESAMEINTO")
print("_______________________________________________________")
print("verificamos si existen datos vacios")
print(dataset.info(verbose=True,null_counts=True))
print("_______________________________________________________")
print("en caso de encontrar con valores vacios se reemplaza por 0")
dataset=dataset.replace(np.nan,"0")
print(dataset.info(verbose=True,null_counts=True))

#creacion de la red neuronal

#alimentacion de valores a comparar y pronosticar
#X=dataset[['LOCATION_ID','TOTAL','numbers','Money_Value','PROB','CONTROL_RISK']]
#X=dataset[['LOCATION_ID','TOTAL']]
X=dataset.loc[:,['TOTAL','numbers','Money_Value','PROB','CONTROL_RISK']]
Y=dataset["LOCATION_ID"]
print("_______________________________________________________")
print("verificamos los valores unicos de la tablas")
print(Y.unique())

#cambiando valor de texto a numero
Y=Y.replace({"LOHARU":'0',"NUH":'0',"SAFIDON":'0'})
print("_______________________________________________________")
print("verificamos los valores unicos de la tablas")
print(Y.unique())
print("_______________________________________________________")

#normalizando los datos


#creamos el modelo de entrenamiento y prueba
#X_test con 80 % de entrenamiento
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20)

#escalando la funciones de entrenamiento
from sklearn.preprocessing import StandardScaler
escalaX=StandardScaler()

X_train=escalaX.fit_transform(X_train)
X_test=escalaX.transform(X_test)

#escalaY=StandardScaler()
#y_train=escalaY.fit_transform(y_train)

#entrenamiento y predicciones de la red neuronal
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,y_train)
prediccion=mlp.predict(X_test)

#evaluacion del algoritmo
from sklearn.metrics import confusion_matrix
print("matriz de confusion")
print(confusion_matrix(y_test,prediccion))
from sklearn.metrics import classification_report
print("_______________________________________________________")
print("reporte de clasificacion")
print(classification_report(y_test,prediccion))

from sklearn.metrics import plot_confusion_matrix
print("_______________________________________________________")
print("grafica de la matriz de confusion")
plot_confusion_matrix(mlp,X_test,y_test,cmap=plt.cm.Blues)


 




