# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:26:52 2025

@author: fjose
"""

import pandas as pd
import itertools 
from tabulate import tabulate

#%% Generar un diseño Factorial General y aleatorizar. 

"""
Construir un diseño factorial general para el siguiente experimento. 
Se desea saber si la concentración de H_2 O_2, NaOH, y el tiempo, 
tienen efecto sobre el índice de blancura de un material lignocelulósico
en un proceso de blanqueamiento.

Factor 1: Concentración H_2 O_2: 10% y 20% 
Factor 2: Concentración de NaOH: 1%, 2.4% y 5%
Factor 3: Tiempo: 1h, 3h, y 5h  
Número de réplicas 3

"""

#factores con sus niveles 

factor1 = [10,20] #H2O2
factor2 =[1,2.4,5] #NaOH
factor3 =[1,3,5] #Tiempo

"""
hacer las combinaciones de los factores, es un conteo de multiples pasos 
en el primer paso hay 2 posibles resultados, en el segundo 3
y en el tercero 3, entonces 2x3x3
usamos itertools.product para hacer esas combinaciones
"""

diseño_F=list(itertools.product(factor1,factor2,factor3))
#al general df multiplicamos la lista por 3 para tener 3 replicas
factorial = pd.DataFrame(diseño_F*3, columns=['H2O2','NaOH','Tiempo'])

#pero hay que aleatorizar las corridas

factorial_random= factorial.sample(frac=1,ignore_index=True)

print(tabulate(factorial,headers='keys',tablefmt='heavy_grid'))

print(tabulate(factorial_random,headers='keys',tablefmt='heavy_grid'))

factorial_random.to_excel("diseño_factorial_general_random.xlsx")

#%%