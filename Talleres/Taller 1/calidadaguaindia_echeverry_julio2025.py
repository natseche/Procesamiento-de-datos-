# -*- coding: utf-8 -*-
# ============================================================
# Taller 1 – Predicción de la Calidad del Agua en la India
# Autor: Natalia Echeverry Salcedo
# Fecha de Inicio: 28/07/2025
# Fecha Actual: 10/08/2025
# ============================================================

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pylab import *

# Bibliotecas de PySpark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator

# Inicialización del contexto y sesión de Spark
contexto = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()
context = SQLContext(contexto)

# ============================================================
# 2. CARGA DE DATOS DESDE GOOGLE DRIVE
# ============================================================

from google.colab import drive
drive.mount("/content/drive/")

rutaCSV = "/content/drive/MyDrive/Colab Notebooks/CalidadAgua/waterquality.csv"
rutaMapa = "/content/drive/MyDrive/Colab Notebooks/CalidadAgua/Indian_States.shp"

os.environ['SHAPE_RESTOR_SHX'] = 'YES'

# Cargar datos CSV con Spark
df00 = spark.read.csv(rutaCSV, header=True, inferSchema=True)

# Vista previa de los primeros 5 registros
df00.show(5)

# ============================================================
# 3. ANÁLISIS Y PREPARACIÓN DE DATOS
# ============================================================

# Listado de columnas
print(df00.columns)

# Estadísticas descriptivas por columna
for valor in df00.columns:
    df00.describe([valor]).show()

# Comprobación de valores nulos o NaN
df00.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) 
             for c in df00.columns]).show()

# Crear vista temporal para SQL en Spark
df00.createOrReplaceTempView("df00_sql")

# Filtrar registros sin valores nulos en columnas clave
df01 = spark.sql('''
    SELECT * FROM df00_sql 
    WHERE TEMP IS NOT NULL
      AND DO IS NOT NULL
      AND pH IS NOT NULL
      AND CONDUCTIVITY IS NOT NULL
      AND BOD IS NOT NULL
      AND NITRATE_N_NITRITE_N IS NOT NULL
      AND FECAL_COLIFORM IS NOT NULL
''')

# Revisión de valores nulos tras filtrado
df01.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) 
             for c in df01.columns]).show()

# ============================================================
# 4. TRATAMIENTO DE TIPOS DE DATOS
# ============================================================

# Conversión de columnas numéricas a tipo Float
df00 = df00.withColumn('TEMP', df00['TEMP'].cast(FloatType())) \
           .withColumn('pH', df00['pH'].cast(FloatType())) \
           .withColumn('DO', df00['DO'].cast(FloatType())) \
           .withColumn('CONDUCTIVITY', df00['CONDUCTIVITY'].cast(FloatType())) \
           .withColumn('NITRATE_N_NITRITE_N', df00['NITRATE_N_NITRITE_N'].cast(FloatType())) \
           .withColumn('FECAL_COLIFORM', df00['FECAL_COLIFORM'].cast(FloatType())) \
           .withColumn('BOD', df00['BOD'].cast(FloatType()))

# Eliminación de columna innecesaria
df01 = df00.drop('TOTAL_COLIFORM')

# ============================================================
# 5. CREACIÓN DE TABLAS PARA VISUALIZACIÓN
# ============================================================

df01.createOrReplaceTempView("df01_sql")

# Función auxiliar para extraer columna como lista
def obtener_parametro(nombre_columna):
    return spark.sql(f"SELECT {nombre_columna} FROM df01_sql") \
                .rdd.map(lambda fila: getattr(fila, nombre_columna)).collect()

# Extracción de parámetros
do_parametro = obtener_parametro("DO")
PH_parametro = obtener_parametro("pH")
COND_parametro = obtener_parametro("CONDUCTIVITY")
BOD_parametro = obtener_parametro("BOD")
NN_parametro = obtener_parametro("NITRATE_N_NITRITE_N")
FC_parametro = obtener_parametro("FECAL_COLIFORM")

# ============================================================
# 6. VISUALIZACIÓN DE PARÁMETROS
# ============================================================

# Gráfico DO y pH
plt.figure(figsize=(13,7))
plt.plot(range(len(do_parametro)), do_parametro, label='Oxígeno Disuelto')
plt.plot(range(len(PH_parametro)), PH_parametro, label='pH')
plt.title('Parámetro DO y pH de la calidad del agua')
plt.legend()
plt.grid()
plt.show()

# Gráfico DO y Nitratos
plt.figure(figsize=(13,7))
plt.plot(range(len(BOD_parametro)), BOD_parametro, label='BOD')
plt.plot(range(len(NN_parametro)), NN_parametro, label='Nitratos')
plt.title('Parámetro BOD y Nitratos de la calidad del agua')
plt.legend()
plt.grid()
plt.show()

# Gráfico Conductividad y Coliformes Fecales
plt.figure(figsize=(13,7))
plt.plot(range(len(COND_parametro)), COND_parametro, label='Conductividad')
plt.plot(range(len(FC_parametro)), FC_parametro, label='Coliformes Fecales')
plt.title('Conductividad y Coliformes Fecales de la calidad del agua')
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 7. CREACIÓN DE COLUMNAS DE CALIDAD SEGÚN RANGOS
# ============================================================

df02 = df01.withColumn("qrPH", F.when((df01.pH>=7) & (df01.pH<=8.5), 100)
                               .when(((df01.pH>=6.8) & (df01.pH<6.9)) | ((df01.pH>8.5) & (df01.pH<8.6)), 80)
                               .when(((df01.pH>=6.7) & (df01.pH<6.8)) | ((df01.pH>=8.6) & (df01.pH<8.8)), 60)
                               .when(((df01.pH>=6.5) & (df01.pH<6.7)) | ((df01.pH>=8.8) & (df01.pH<9.0)), 40)
                               .otherwise(0))

df02 = df02.withColumn("qrDO", F.when((df01.DO>=6), 100)
                               .when((df01.DO>=5.1) & (df01.DO<6.0), 80)
                               .when((df01.DO>=4.1) & (df01.DO<5.0), 60)
                               .when((df01.DO>=3.0) & (df01.DO<=4.0), 40)
                               .otherwise(0))

df02 = df02.withColumn("qrCOND", F.when((df01.CONDUCTIVITY>=0.0) & (df01.CONDUCTIVITY<=75.0), 100)
                                  .when((df01.CONDUCTIVITY>75.0) & (df01.CONDUCTIVITY<=105.0), 80)
                                  .when((df01.CONDUCTIVITY>150.0) & (df01.CONDUCTIVITY<=225.0), 60)
                                  .when((df01.CONDUCTIVITY>225.0) & (df01.CONDUCTIVITY<=300.0), 40)
                                  .otherwise(0))

df02 = df02.withColumn("qrBOD", F.when((df01.BOD>=0.0) & (df01.BOD<3.0), 100)
                                 .when((df01.BOD>=3.0) & (df01.BOD<6.0), 80)
                                 .when((df01.BOD>=6.0) & (df01.BOD<80.0), 60)
                                 .when((df01.BOD>=80.0) & (df01.BOD<125.0), 40)
                                 .otherwise(0))
