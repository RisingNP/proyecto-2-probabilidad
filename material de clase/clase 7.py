# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:48:30 2025

@author: fjose
"""

import numpy as np 
import pandas as pd 
#import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest


#%% Importar datos 

df = pd.read_excel("visceral_fat.xlsx")

#si la variable tiene NaN las pruebas no arrojaran un resultado, se deben imputar esos datos 
# por ejemplo con la media así

#en al bmi hay un NAN entonces 
df['age (years)'].fillna(df['age (years)'].mean(),inplace=True)
#%% Funciones para calcular las pruebas 

def t_test_one(data,mu,variable):
    print(f"Prueba T para una sola muestra para {variable}")
    t_stat, p_value = stats.ttest_1samp(data, mu)
    print(f"Estadístico = {t_stat:.4f}, valor_p = {p_value:.4f}")

    
def proportions(exitos,n,proportion,variable,direccion):
    #direccion: "two-sided", "larger", "smaller" 
    print(f"Prueba de proporciones para {variable}")
    stat, p_value = proportions_ztest(count=exitos, nobs=n, value=proportion, alternative='two-sided')
    print(f"Estadístico = {stat:.4f}, Valor_p = {p_value:.4f}")

# Pruebas de normalidad
def test_normalityKS(data, variable):
    print(f"\n Análisis de normalidad por Kolmogorov-Smirnov para '{variable}'")

    # Kolmogorov-Smirnov (KS) test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f" Estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4f}")

def test_normalitySW(data, variable):
    print(f"\n Análisis de normalidad por Shapiro-Wilk para '{variable}'")
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")



