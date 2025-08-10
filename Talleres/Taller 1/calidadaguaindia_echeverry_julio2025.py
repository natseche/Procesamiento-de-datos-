# -*- coding: utf-8 -*-
"""##**1. Importación de bibliotecas**"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

#Bibliotecas especializadas
from pylab import *


#Bibliotecas de Contexrp PySpark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F #Acceso a todas las funciones

contexto = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

from pyspark.sql import SparkSession
##Se crea la a¿sesión para el tratamiento de alto volúmen de datos: calidad del agua
spark = SparkSession.builder.getOrCreate()

context =SQLContext(contexto)

from pyspark.ml.evaluation import RegressionEvaluator

"""##**2. Carga de datos desde el IDRIVE**"""

#Se conecta con IDRIVE de google
from google.colab import drive
drive.mount("/content/drive/")

rutaCSV = "/content/drive/MyDrive/Colab Notebooks/CalidadAgua/waterquality.csv"
rutaMapa = "/content/drive/MyDrive/Colab Notebooks/CalidadAgua/Indian_States.shp"

os.environ['SHAPE_RESTOR_SHX'] = 'YES'

#Se empieza a cargar los
df00 = spark.read.csv(rutaCSV, header=True, inferSchema=True)
#se requiere observar los primeros 5 datos cabecera
df00.show(5)

"""##**3. Análisis y Preparación de Datos**
Se presentan datos en formato CSV con los diferentes parámetros de los rios de la India. Cada paramentro es el promedio de los valores medidos en un período de tiempo. Los datos han sido extraidos de la web oficial de la India (RiverIndia)

-Tipo de datos: coherencia de tipo de datos, transformación.

-Columnas: conocimiento de las columnas, eliminación.

-Analisis de datos nulos o imposibles: conocimiento

"""

#Conocimiento de las columnas
df00.columns

"""* STATION CODE: Código de estación de medida
* LOCATIONS
* STATE
* TEMP
* DO
* pH
* CONDUCTIVITY
* BOD
* NITRATE_N_NITRITE_N
* FECAL_COLIFORM: Promedio de bacterias coliformes: excresiones
* TOTAL_COLIFORM: Se eliminara pues no aporta al estuido de prediccion de calidad del agua

**Se presentan las estadísticas de datos a continuación:**
"""

for valor in df00.columns:
  df00.describe([valor]).show()

"""##**Visualización de los Datos**"""

#Se cres una tabla de

df00.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df00.columns]).show()

"""

*   Se observa que no hay datos nulos o imposibles
*   Se procede a graficar cada una de las dimensiones

"""

#Se crea una vista para ser usada en las visualizaciones

df00.createOrReplaceTempView("df00_sql")

df01 = spark.sql('''Select * from df00_sql where TEMP is not null and
            DO is not null and
            pH is not null and
            CONDUCTIVITY is not null and
            BOD is not null and
            NITRATE_N_NITRITE_N is not null and
            FECAL_COLIFORM is not null''')

#Se verifica la Cantidad de Valores Nulos o Imposibles

df01.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df01.columns]).show()

"""**Tratamiento de datos**"""

df00.dtypes

#Se procede a cambiar los tipos de datos
df00=df00.withColumn('TEMP', df00['TEMP'].cast(FloatType()))
df00=df00.withColumn('pH', df00['pH'].cast(FloatType()))
df00=df00.withColumn('DO', df00['DO'].cast(FloatType()))
df00=df00.withColumn('CONDUCTIVITY', df00['CONDUCTIVITY'].cast(FloatType()))
df00=df00.withColumn('NITRATE_N_NITRITE_N', df00['NITRATE_N_NITRITE_N'].cast(FloatType()))
df00=df00.withColumn('FECAL_COLIFORM', df00['FECAL_COLIFORM'].cast(FloatType()))
df00=df00.withColumn('BOD', df00['BOD'].cast(FloatType()))
df00.dtypes

#Se elimina la columna de TOTAL_COLIFORM
df01 = df00.drop('TOTAL_COLIFORM')
df01.columns

"""**Creación de Tablaas para Visualizar los datos**


*   Se hace uso de la función LAMBDA para hacer el tratamiento y limpieza de los datos


"""

df01.createOrReplaceTempView("df01_sql")

### Se crea una consulta por cada parametro
do_parametro = spark.sql("Select DO from df01_sql")

### rdd : Resilient Distributed Dataset :: son los datos pero ya de manera distribuida

### Se hace una consulta para crear el vector de la tabla por cada parametro
do_parametro= do_parametro.rdd.map(lambda fila: fila.DO).collect()

### con PH
PH_parametro = spark.sql("Select pH from df01_sql")
PH_parametro= PH_parametro.rdd.map(lambda fila: fila.pH).collect()

### con CONDUCTIVITY (COND)
COND_parametro = spark.sql("Select CONDUCTIVITY from df01_sql")
COND_parametro= COND_parametro.rdd.map(lambda fila: fila.CONDUCTIVITY).collect()

### con BOD
BOD_parametro = spark.sql("Select BOD from df01_sql")
BOD_parametro= BOD_parametro.rdd.map(lambda fila: fila.BOD).collect()

### con NN
NN_parametro = spark.sql("Select NITRATE_N_NITRITE_N from df01_sql")
NN_parametro= NN_parametro.rdd.map(lambda fila: fila.NITRATE_N_NITRITE_N).collect()

### con CF
FC_parametro = spark.sql("Select FECAL_COLIFORM from df01_sql")
FC_parametro= FC_parametro.rdd.map(lambda fila: fila.FECAL_COLIFORM).collect()

# Grafica de los parametros para conocer sus caracteristicas
tam = len(do_parametro)
fig, ax1 = plt.subplots(num=None,  figsize=(13,7), facecolor='w', edgecolor='k')
ax1.plot(range(0, tam), do_parametro, label='Oxigeno Disuelto')
ax1.plot(range(0, tam), PH_parametro, label='pH')
fig.suptitle('Parámetro DO y ph de la calidad del Agua')
legend=ax1.legend()
plt.grid()
plt.show()

# Grafica de los parametros para conocer sus caracteristicas
tam = len(do_parametro)
fig, ax1 = plt.subplots(num=None,  figsize=(13,7), facecolor='w', edgecolor='k')
ax1.plot(range(0, tam), BOD_parametro, label='Oxigeno Disuelto')
ax1.plot(range(0, tam), NN_parametro, label='pH')
fig.suptitle('Parámetro DO y ph de la calidad del Agua')
legend=ax1.legend()
plt.grid()
plt.show()

# Grafica de los parametros para conocer sus caracteristicas
tam = len(do_parametro)
fig, ax1 = plt.subplots(num=None,  figsize=(13,7), facecolor='w', edgecolor='k')
ax1.plot(range(0, tam), COND_parametro, label='Conductividad')
ax1.plot(range(0, tam), FC_parametro, label='CF_parametro')
fig.suptitle('Parámetro Conductividad y Material Fecal de la calidad del Agua')
legend=ax1.legend()
plt.grid()
plt.show()

#Se requiere hacer una funcion definida por el usuario: que permita definir el rango de la calidad del agua segun el PH
#Se crea una columna para los rangos del parametro

df02 = df01.withColumn("qrPH", F.when((df01.pH>=7)& (df01.pH<=8.5), 100).
                                when(((df01.pH>=6.8) & (df01.pH<6.9)) | ((df01.pH>8.5) & (df01.pH<8.6)), 80).
                                when(((df01.pH>=6.7) & (df01.pH<6.8)) | ((df01.pH>=8.6) & (df01.pH<8.8)), 60).
                                when(((df01.pH>=6.5) & (df01.pH<6.7)) | ((df01.pH>=8.8) & (df01.pH<9.0)), 40).otherwise(0))

#Funcion definida por el usuario para definir el rando de la calidad del agua segun Do
df02 = df02.withColumn("qrDO", F.when((df01.DO>=6), 100).
                      when((df01.DO>=5.1) & (df01.DO<6.0),80).
                      when((df01.DO>=4.1) & (df01.DO<5.0), 60).
                      when((df01.DO>=3.0) & (df01.DO<=4.0), 40).otherwise(0))

df02 = df01.withColumn("qrCOND", F.when((df01.CONDUCTIVITY>=0.0) & (df01.CONDUCTIVITY<=75.0), 100).
                                when((df01.CONDUCTIVITY>75.0) & (df01.CONDUCTIVITY<=105.0), 80).
                                when((df01.CONDUCTIVITY>150.0) & (df01.CONDUCTIVITY<=225.0), 60).
                                when((df01.CONDUCTIVITY>225.0) & (df01.CONDUCTIVITY<=300.0), 40).otherwise(0))

df02 = df01.withColumn("qrBOD", F.when((df01.BOD>=0.0) & (df01.BOD<3.0), 100).
                                when((df01.BOD>=3.0) & (df01.BOD<6.0), 80).
                                when((df01.BOD>=6.0) & (df01.BOD<80.0), 60).
                                when((df01.BOD>=80.0) & (df01.BOD<125.0), 40).otherwise(0))
