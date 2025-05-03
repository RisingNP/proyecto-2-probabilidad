# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:48:33 2025

@author: fjose
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats 
from statsmodels.sandbox.stats.runs import runstest_1samp # para la prueba Runs 
from statsmodels.formula.api import ols #modelo lineal del ANOVA
import statsmodels.api as sm 
from tabulate import tabulate
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%% Funciones necesarias. 

def test_normalityKS(data, variable): # Pruaba de Normalidad Kolmogorov-Smirnof 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """  
    print(f"\n Análisis de normalidad por Kolmogorov-Smirnov para '{variable}'")

    # Kolmogorov-Smirnov (KS) test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f" Estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4f}")

def test_normalitySW(data, variable): # Prueba de Normalizas Shapiro-Wilks 
    """
    data: arreglo de datos a evaluar la normalidad
    variable: string con el nombre de la variable 
    """
    print(f"\n Análisis de normalidad por Shapiro-Wilk para '{variable}'")
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")
    
def random_test(residuos):
    """
    Parameters
    ----------
    residuos : Array
        DESCRIPTION: Residuos del ANOVA 

    Returns
    -------
    None.

    """
    _, p_runs = runstest_1samp(residuos, correction=True)

    print(f"Prueba de Runs: p-valor={p_runs}")
    
def test_homogeneityL(var1, var2, name1, name2): # Prueba de levene
    """
    var1 y var2: variables a las que se corroborará homocedasticidad 
    name1 y name2: strings con el nnombre de las variables
    """
    print(f"\n Análisis de homocedasticidad entre '{name1}' y '{name2}'")

    # Prueba de Levene (no asume normalidad)
    levene_stat, levene_p = stats.levene(var1, var2)
    print(f"Levene test: Estadístico = {levene_stat:.4f}, p-valor = {levene_p:.4f}")

def t_test_one(data,mu,variable): #Prueba T para una muestra
    """
    data: arreglo de datos a comparar
    mu: media poblacional o valor de referencia 
    variable: string con el nombre de la variable que se está comparando
    """
    print(f"Prueba T para una sola muestra para {variable}")
    t_stat, p_value = stats.ttest_1samp(data, mu)
    print(f"Estadístico = {t_stat:.4f}, valor_p = {p_value:.4f}")
    
def box_cox(data): #transformación depotencia   

    transformed_data, lambda_opt = stats.boxcox(data)
    return transformed_data, lambda_opt

def tukey(respuesta,factor, alfa,n_factor):
    """

    Parameters
    ----------
    respuesta : Array
        DESCRIPTION. Array con los datos de la variable respuesta
    factor : Array
        DESCRIPTION.Array con los niveles del factor 
    alfa : Float
        DESCRIPTION. Valor alfa de comparación 
    n_factor : String
        DESCRIPTION. Nombre del factor

    Returns
    -------
    None.

    """
    
    tukey = pairwise_tukeyhsd(respuesta, factor, alpha=alfa)
    print(f"Prueba Tukey para el factor {n_factor}")
    print(tukey)
    
def kruskal_W(df,Respuesta,Factor):
    """
    
    Parameters
    ----------
    df : Data_Frame
        DESCRIPTION. estructura con los datos del experimento
    Respuesta : String
        DESCRIPTION. nombre de la variable respuesta, key del dataframe
    Factor : String
        DESCRIPTION. nombre del factor, key del dataframe

    Returns
    -------
    None.

    """
    grupos_B = [df[Respuesta][df[Factor] == nivel] for nivel in df[Factor].unique()]
    stat_B, p_B = stats.kruskal(*grupos_B)
    print(f"Kruskal-Wallis para {Factor}: H = {stat_B:.4f}, p = {p_B:.4f}")
    
    
def kruskal_interaccion(df,Respuesta,Factor1,Factor2):
    """
    

    Parameters
    ----------
    df : Data_Frame
        DESCRIPTION. estructura con los datos del experimento
    Respuesta : String
        DESCRIPTION. nombre de la variable respuesta, key del dataframe
    Factor1 : String
        DESCRIPTION. nombre del factor1, key del dataframe
    Factor2 : String
        DESCRIPTION.nombre del factor12, key del dataframe

    Returns
    -------
    None.

    """
    
    df['interaccion'] = df[Factor1].astype(str) + "_" + df[Factor2].astype(str) # se genera una columana con las combinaciones entre factores

    grupos_interaccion = [df[Respuesta][df['interaccion'] == nivel] for nivel in df['interaccion'].unique()]
    stat_int, p_int = stats.kruskal(*grupos_interaccion)
    print(f"Kruskal-Wallis para la interacción {Factor1}x{Factor2} p = {p_int:.4f}")
  
#%% Cargar datos 
df = pd.read_excel("3 factores.xlsx")

#%% Supuesto de normalidad de la variable respuesta

test_normalitySW(df['conductividad'],'Variable Respuesta')

#si no fuera normal haríamos una tranformación
data_t, lambda_opt = box_cox(df['conductividad'])

test_normalityKS(data_t,'Variable Respuesta transformada')

#lambda_opt = 1# si no se hizo tranformación, lambda = 1
#%% Supuesto de Homocedasticidad

#%% factor 1 acido
#nivel 1
nivel1=np.power(df[df['acido']==0]['conductividad'],lambda_opt)
nivel2=np.power(df[df['acido']==6]['conductividad'],lambda_opt)
nivel3=np.power(df[df['acido']==12]['conductividad'],lambda_opt)
nivel4=np.power(df[df['acido']==18]['conductividad'],lambda_opt)
_, levene_p = stats.levene(nivel1,nivel2,nivel3,nivel4)

print(f"Levene test: p-valor = {levene_p:.4f}")

#%% factor 2 sal
nivel1=np.power(df[df['sal']==0]['conductividad'],lambda_opt)
nivel2=np.power(df[df['sal']==10]['conductividad'],lambda_opt)
nivel3=np.power(df[df['sal']==20]['conductividad'],lambda_opt)

_, levene_p = stats.levene(nivel1,nivel2,nivel3)
_, bartlett_p = stats.bartlett(nivel1,nivel2,nivel3)
print(f"Levene test: p-valor = {levene_p:.4f}")
print(f"Bartlett test: p-valor = {bartlett_p:.4f}")

#%% factor 3 temp

nivel1=np.power(df[df['temperatura']==80]['conductividad'],lambda_opt)
nivel2=np.power(df[df['temperatura']==100]['conductividad'],lambda_opt)

_, levene_p = stats.levene(nivel1,nivel2)

print(f"Levene test: p-valor = {levene_p:.4f}")

#%% Graficos de interacción 

fig, ax = plt.subplots(figsize=(8,5))

sns.pointplot(data=df, x="sal", y="conductividad", hue="acido",
              dodge=True, markers=["o", "s"], linestyles=["-", "--"],
              palette="tab10", ci=95)

plt.title("Gráfico de Interacción entre Factor_A y Factor_B")
plt.xlabel("Factor A")
plt.ylabel("Media de la Respuesta")
plt.legend(title="Factor B")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

#%% Realizar el ANOVA 

df['conductividad'] = np.power(df['conductividad'],lambda_opt)
# Los ANOVAS son modelos lineales, entonces se debe ajustar el modelo
#previa inspección de las interacciones uno supone si debe incluiras
# o puede hacerlo siempre y luego retirarlas segun su siignificancia
#modelo = ols('conductividad ~ C(acido)*C(sal)*C(temperatura)', data=df).fit()
#modelo = ols('conductividad ~ C(acido)+C(sal)+C(temperatura)+C(acido):C(sal)+C(acido):C(temperatura)+C(temperatura):C(sal)+C(sal):C(temperatura):C(acido)', data=df).fit()
modelo = ols('conductividad ~ C(acido)+C(sal)', data=df).fit()


anova_table = sm.stats.anova_lm(modelo, typ=3)


# Mostrar resultados
print(tabulate(anova_table,headers='keys',tablefmt='heavy_grid'))

#%% extracción de residuos 

df['Residuos']=modelo.resid

# normalidad en los residuos 

test_normalityKS(df['Residuos'],'Residuos')

t_test_one(df['Residuos'],0,"Residuos")

random_test(df['Residuos'])

sm.qqplot(df['Residuos'], line='45', fit=True)
plt.title("QQ-Plot de los Residuos del ANOVA")
plt.show()

"""se comprueba los supuestos, entonces el anova es valido 
y se puede afirmar que solo la concentración del acido y la sal tinene 
efecto sobre la conductividad del metal que es limpiado. 
"""

#%% comparaciones multiples 
"""
Como ya sabemos con certeza que la concentración de acido y sal
tiene efecto sobre la conductividad del metal qeu se limpia
y además los factores de acido y sal tienen 3 o más niveles 
se debe hacer una prueba de comparaciones multiple para determinar
si hay diferencia entre los niveles de cada factor 
"""

# Prueba de Tukey para acido 
tukey(df["conductividad"], df["acido"],0.05,"Ácido")

#prueba Tukey para sal 

tukey(df["conductividad"], df["sal"],0.05,"Sal")

#%% Kruskal Wallis Test 

"""
supongamos que el anterior ejercicio no cumplió con los requisitos para
un ANOVA, entonces se debe hacer una prueba de Kruskal Wallis
Como las pruebas no paramétricas son robustas ante las distribuciones
aunque el ejercicio anterior cumplió en general con todo, el resultado 
de esta prueba debería ser similar a lo ya obtenido.
"""
#para el acido
kruskal_W(df,'conductividad','acido') # notese que se usa el df sin tranformar 

#para la sal
kruskal_W(df,'conductividad','sal')

#para la temperatura
kruskal_W(df,'conductividad','temperatura')

#%%
"""
El resultado es similar, la temperatura no tiene efecto significativo, 
mientras que el acido y la sal sí. Ahora bien, no podemos saber con esta prueba 
si existen interacciones. Podríamos recurrir al gráfico para saberlo, pero se puede
evaluar una a una las interacciones 
"""

kruskal_interaccion(df,'conductividad','acido','sal')

"""
Segun este resultado, la interacción sería significativa, pero no observamos eso en el 
gráfico, y tampoco se observó en el ANOVA, este podría ser un falso positivo. 
la prueba kruskal wallis tiene como condición que haya al menos 5 datos en cada grupo y hay 4

"""