 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Moreno Valero
# NOMBRE: Juan Miguel
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Rodríguez Páez
# NOMBRE: Manuel Jesús
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    

# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


## ---------- 




def particion_entr_prueba(X, y, test=0.20):
    i_c = {}
    for clase in set(y):
        i_c[clase] = [i for i, c in enumerate(y) if c == clase]

    i_train = []
    i_test = []
    for clase_i in i_c.values():
        np.random.shuffle(clase_i)
        n_prueba = int(len(clase_i) * test)
        i_train.extend(clase_i[:-n_prueba]) 
        i_test.extend(clase_i[-n_prueba:])

    l_train = [X[i] for i in i_train]
    l_test = [X[i] for i in i_test]
    li_train = [y[i] for i in i_train]
    li_test = [y[i] for i in i_test]

    X_train = np.array(l_train)
    X_test = np.array(l_test)
    y_train = np.array(li_train)
    y_test = np.array(li_test)

    return X_train, X_test, y_train, y_test




# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


#class NormalizadorStandard():

#    def __init__(self):
#       ...
        
#    def ajusta(self,X):
#       ...

#    def normaliza(self,X):
#       ...

 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    

# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# <<< error_cancer=NormalizadorStandard()
# <<< error_cancer.normaliza(Xe_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 


class NormalizadorStandard():

    def __init__(self):
        self.media = None
        self.std = None
        self.ajustado = False
        
    def ajusta(self,X):
        self.media = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)   
        self.ajustado = True

    def normaliza(self,X):
        if (self.ajustado==False):
            raise NormalizadorNoAjustado("Debes llamar al método de normalización antes de ajustar el normalizador")
        return (X - self.media) / self.std


#np.mean(Xe_cancer_n)
#np.std(Xe_cancer_n)



# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 


class NormalizadorMinMax():
    
    def __init__(self):
        self.min = None
        self.max = None
        self.ajustado = False
        
    def ajusta(self,X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)   
        self.ajustado = True

    def normaliza(self,X):
        if (self.ajustado==False):
            raise NormalizadorNoAjustado("Debes llamar al método de normalización antes de ajustar el normalizador")
        return (X - self.min) / (self.max - self.min)



#np.mean(Xe_cancer_n)
#np.std(Xe_cancer_n)
#np.mean(Xv_cancer_m)
#np.std(Xv_cancer_m)
#np.mean(Xp_cancer)
#np.std(Xp_cancer)


# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

from scipy.special import expit    

def sigmoide(x):
    return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])




# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------



class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
                 batch_tam=64):
        self.rate=rate
        self.rate_decay=rate_decay
        self.n_epochs=n_epochs
        self.batch_tam=batch_tam
        self.clase=None
        self.ajustado=False
        self.w=None
        
        
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
                     early_stopping=False,paciencia=3):
        
        self.ajustado=True
        negativa= np.unique(y)[0]
        positiva=np.unique(y)[1]
        self.clase = [negativa, positiva] 
        
        # Inicialización de la matriz de pesos aleatorios    
        self.w = [np.random.uniform(-1,1) for _ in np.arange(X.shape[1]+1)]
        
        entropia_cruzada = float('inf')
        cont=1
        rate=self.rate
        
        for e in np.arange(n_epochs):                  
            
            if(self.rate_decay==True):
                self.rate = rate*(1/(1+e))
                rate = self.rate
                
            entropia_cruzada_entr, self.w = epoch(self.w,X,y,salida_epoch,early_stopping,self.batch_tam, rate)
            rend = rendimiento(self,X,y)            
            if(salida_epoch==True):
                print("\nEpoch " + str(e) + ", en entrenamiento EC: " + str(entropia_cruzada_entr) + "," +
                      "rendimiento: " + str(rend))
            
            if(early_stopping==True):
                entropia_cruzada_val, self.w = epoch(self.w,Xv,yv,salida_epoch,early_stopping,self.batch_tam, rate)
                rend = rendimiento(self,X,y)                 
                if(salida_epoch==True):
                    print("         en validación    EC: " + str(entropia_cruzada_val) + "," +
                          "rendimiento: " + str(rend))
                    
                
                if(entropia_cruzada<=entropia_cruzada_val):
                    cont+=1
                    
                    if(cont==paciencia):
                        print("\nPARADA TEMPRANA")
                        break
                
                else:
                    cont=1
                    
                    
                entropia_cruzada=entropia_cruzada_val
                
                


    def clasifica_prob(self,ejemplos):
        if (self.ajustado==False):
            raise ClasificadorNoEntrenado("El clasificador aún no ha sido entrenado")
            
        ypred=np.array([])
        bias = self.w[0]
        for i in np.arange(ejemplos.shape[0]):
            combLineal = np.dot(ejemplos[i], self.w[1:]) + bias
            ypred = np.append(ypred, sigmoide(combLineal))
            
            
        return ypred
    

    
    def clasifica(self,ejemplo):
        if (self.ajustado==False):
            raise ClasificadorNoEntrenado("El clasificador aún no ha sido entrenado")
            
        yprob = self.clasifica_prob(ejemplo)
        ypred = np.array([], dtype=int)
        for p in np.arange(len(yprob)):
            if(yprob[p]<0.5):
                ypred = np.append(ypred, self.clase[0])
                
            else:
                ypred = np.append(ypred, self.clase[1])
                
        return ypred
    


    
def epoch(w,X,y,salida_epoch,early_stopping,batch_tam,rate):
    
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)   
    bias = w[0]
    
    
    for b in np.arange(0, X.shape[0], batch_tam):
           
        ind_batch = indices[b:b+batch_tam]
        combLineal = np.dot(X[ind_batch], w[1:]) + bias
        ypred = sigmoide(combLineal)
        bact = np.sum(y[ind_batch]-ypred)
        biasAct = bias + rate*bact
        minibatch = np.dot(X[ind_batch].T,(y[ind_batch]-ypred))
        pesosAct = w[1:] + rate*minibatch
        w = np.concatenate((np.array([biasAct]), pesosAct))
        
        
    combLineal = np.dot(X, w[1:]) + bias
    ypred = sigmoide(combLineal)
    entropia_cruzada = 0.
    if(salida_epoch==True or early_stopping==True):
         entropia_cruzada = np.sum(np.where(y==1, -np.log(ypred), -np.log(1-ypred)))
    

        
        
    return entropia_cruzada, w 





# ------------------------------------------------------------------------------





# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este ejercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"batch_tam":16,"rate":0.01,"rate_decay":True}, Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------


def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5):
    
    # CREAMOS PARTICIONES ALEATORIAS Y ESTRATIFICADAS     
    i_c = {}
    for clase in set(y):
        i_c[clase] = [i for i, c in enumerate(y) if c == clase]
        
    
    clases = []
    for clase_i in np.arange(len(set(y))):
        tamaño = len(i_c[clase_i])
        cl = i_c[clase_i]
        np.random.shuffle(cl)   # Barajar los índices de cada clase
        cl = cl[:-(tamaño%n)] #Reducimos el tamaño de nuestras clases para obtener particiones del mismo tamaño
        clases.append(cl) 
        
        
    folds = []
    for j in np.arange(n):
        fold = np.array([], dtype=int)
        for i in np.arange(len(set(y))):
            c = np.array_split(clases[i], n)
            fold = np.concatenate((fold, c[j]))
            
        np.random.shuffle(fold)    #mezclamos los "0s" y los "1s"
        folds.append(fold)
        
    
    # REALIZAMOS LA VALIDACIÓN CRUZADA
    media=0
    
    for i in np.arange(n):
        X_test = np.array([X[j] for j in folds[i]])
        y_test = np.array([y[j] for j in folds[i]])
        
        train = []
        for e in np.arange(n):
            if(e!=i):
                train.append([j for j in folds[e]])
        
        
        train = np.concatenate(train)    #aplanamos
        
        X_train = np.array([X[j] for j in train])
        y_train = np.array([y[j] for j in train])
        
        cl = clase_clasificador(**params)
        cl.entrena(X=X_train, y=y_train)
        rend = rendimiento(cl,X_test,y_test)
        print("Partición: " + str(i+1) + ". Rendimiento: " + str(rend))
        
        media+=rend
    
    print(media/n)
    
    
    

# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------


# normst=NormalizadorStandard()    #Normalizador


# ######## - Votos de congresistas US  #######
# from carga_datos import X_votos, y_votos

# def transformacionDatos(X,y,tiponeg, tipopos):
#     ynew = np.zeros(y.shape[0], dtype=int)
#     for i in np.arange(X.shape[0]):
#         if(y[i]==tipopos):
#             ynew[i] = 1
            
#     return X,ynew
    

# X_votos_transf, y_votos_transf = transformacionDatos(X_votos, y_votos, np.unique(y_votos)[0], np.unique(y_votos)[1])
# Xev_train, Xp_votos, yev_train, yp_votos = particion_entr_prueba(X_votos_transf,y_votos_transf,test=0.2)
# Xe_votos, Xv_votos, ye_votos, yv_votos = particion_entr_prueba(Xev_train,yev_train,test=0.2)
# normst.ajusta(Xe_votos)
# Xe_votos_n=normst.normaliza(Xe_votos)


# # BATCH_TAM
# for b in np.arange(16,64,16):   
#     print("\nValidación cruzda con batch_size=" + str(b))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":b,"rate":0.01,"rate_decay":True}, 
#                                     Xe_votos_n,ye_votos,n=5)
    
# # RATE
# rateVotos = 1/100
# for r in np.arange(3):  
#     print("\nValidación cruzda con rate=" + str(rateVotos))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":16,"rate":rateVotos,"rate_decay":True}, 
#                                     Xe_votos_n,ye_votos,n=5)
#     rateVotos = rateVotos/10
    
    
# # N FOLDS
# print("\nValidación cruzda con n=" + str(10))
# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                 {"batch_tam":16,"rate":0.1,"rate_decay":True}, 
#                                 Xe_votos_n,ye_votos,n=10)

# modeloVotos=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# train = modeloVotos.entrena(Xe_votos_n, ye_votos, salida_epoch=True)


# ######## - Cáncer de mama #######
# from carga_datos import X_cancer, y_cancer

# Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
# Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# normst.ajusta(Xe_cancer)
# Xe_cancer_n=normst.normaliza(Xe_cancer)


# # BATCH_TAM
# for b in np.arange(16,64,16):
#     print("\nValidación cruzda con batch_size=" + str(b))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":b,"rate":0.01,"rate_decay":True}, 
#                                     Xe_cancer_n,ye_cancer,n=5)
    

# # RATE
# rateCancer = 1/100
# for i in np.arange(3):  
#     print("\nValidación cruzda con rate=" + str(rateCancer))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":16,"rate":rateCancer,"rate_decay":True}, 
#                                     Xe_cancer_n,ye_cancer,n=5)
#     rateCancer = rateCancer/10
    
    
# # N FOLDS
# print("\nValidación cruzda con n=" + str(10))
# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                 {"batch_tam":16,"rate":0.1,"rate_decay":True}, 
#                                 Xe_cancer_n,ye_cancer,n=10)
    

# modeloCancer=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# train = modeloCancer.entrena(Xe_cancer_n, ye_cancer, salida_epoch=True)




# # ####### - Críticas de películas en IMDB   #######
# from carga_datos import X_train_imdb, y_train_imdb

# Xe_imdb,Xv_imdb,ye_imdb,yv_imdb=particion_entr_prueba(X_train_imdb,y_train_imdb,test=0.2)
# normst.ajusta(Xe_imdb)
# X_train_imdb_n=normst.normaliza(Xe_imdb)

# # BATCH_TAM
# for b in np.arange(16,64,16):
#     print("\nValidación cruzda con batch_size=" + str(b))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":b,"rate":0.01,"rate_decay":True}, 
#                                     X_train_imdb_n,ye_imdb,n=5)
    
    
# # RATE
# rateIMDB = 1/100
# for i in np.arange(3):
#     print("\nValidación cruzda con rate=" + str(rateIMDB))
#     rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                     {"batch_tam":b,"rate":0.01,"rate_decay":True}, 
#                                     X_train_imdb_n,ye_imdb,n=5)
    
#     rateIMDB = rateIMDB/10
    

# # N FOLDS
# print("\nValidación cruzda con n=" + str(10))
# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, 
#                                 {"batch_tam":16,"rate":0.1,"rate_decay":True}, 
#                                 X_train_imdb_n,ye_imdb,n=10)
    

# modeloIMDB=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# train = modeloIMDB.entrena(X_train_imdb_n, ye_imdb, salida_epoch=True)
    





# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------


class RL_OvR():

    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.classifiers = []

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        self.classifiers = []

        for i in range(n_classes):
            # Crea un clasificador binario para cada clase
            classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)
            # Prepara los datos para el clasificador binario
            binary_y = (y == unique_classes[i]).astype(int)
            # Entrena el clasificador binario
            classifier.entrena(X, binary_y, n_epochs, salida_epoch)
            self.classifiers.append(classifier)

    def clasifica(self, ejemplos):
        prob_matrix = np.zeros((len(ejemplos), len(self.classifiers)))

        for i, classifier in enumerate(self.classifiers):
            # Clasifica los ejemplos con el clasificador binario correspondiente
            prob_matrix[:, i] = classifier.clasifica_prob(ejemplos)

        # Determina la clase con mayor probabilidad para cada ejemplo
        predictions = np.argmax(prob_matrix, axis=1)

        return predictions 



            
# --------------------------------







# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
                    # ["b",2,"c","y"],
                    # ["c",1,"d","x"],
                    # ["a",2,"d","z"],
                    # ["c",1,"e","y"],
                    # ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

# -------- 

def codifica_one_hot(X):
    
    clases = []
    filas = len(X)
    columnas = len(X[0])
    num=[]
    
    for j in np.arange(columnas):
        cl = set()
        for i in np.arange(filas):
            if(X[i][j] not in cl):
                cl.add(X[i][j])
                clases.append(X[i][j])
                
        num.append(len(cl))
                
    onehot = np.zeros((filas,len(clases)))
    
    
    for i in np.arange(filas):
        inicio=0
        rango=0
        tam=0
        for j in np.arange(columnas):
            inicio+=tam
            tam = num[j]
            rango = inicio + tam
            subclase = clases[inicio:rango]
            if(X[i][j] in subclase):
                elemento = subclase.index(X[i][j])
                onehot[i][inicio+elemento] = 1
            
    
    return onehot





# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------


from carga_datos import X_credito, y_credito

X_credito_coh = codifica_one_hot(X_credito)
y_credito_num = np.zeros(y_credito.shape[0], dtype=int)
for i in np.arange(X_credito_coh.shape[0]):
    for j in np.arange(len(np.unique(y_credito))):
        if(y_credito[i]==np.unique(y_credito)[1]):
            y_credito_num[i] = 1
        elif(y_credito[i]==np.unique(y_credito)[2]):
            y_credito_num[i] = 2
            
Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito_coh,y_credito_num)

# Primer clasificador - batch_tam=8
rl_credito_ovr=RL_OvR(rate=0.001,batch_tam=8)
rl_credito_ovr.entrena(Xe_credito,ye_credito)
rendimiento(rl_credito_ovr,Xe_credito,ye_credito)

# Segundo clasificador - batch_tam=16
rl_credito_ovr=RL_OvR(rate=0.001,batch_tam=16)
rl_credito_ovr.entrena(Xe_credito,ye_credito)
rendimiento(rl_credito_ovr,Xe_credito,ye_credito)

# Tercer clasificador - batch_tam=32
rl_credito_ovr=RL_OvR(rate=0.001,batch_tam=32)
rl_credito_ovr.entrena(Xe_credito,ye_credito)
rendimiento(rl_credito_ovr,Xe_credito,ye_credito)

# Cuarto clasificador - rate=0.0001
rl_credito_ovr=RL_OvR(rate=0.0001,batch_tam=32)
rl_credito_ovr.entrena(Xe_credito,ye_credito)
rendimiento(rl_credito_ovr,Xe_credito,ye_credito)

# Quinto clasificador - rate=0.01
rl_credito_ovr=RL_OvR(rate=0.01,batch_tam=32)
rl_credito_ovr.entrena(Xe_credito,ye_credito)
rendimiento(rl_credito_ovr,Xe_credito,ye_credito)


# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------

def lecturaFichero(nombre):
    with open("datos/digitdata/" + nombre , "r") as file:
        content = [line.strip() for line in file.readlines()]
        
    return content


def cargaImagenes(X):
    res= []
    for lines in np.arange(0, len(X), 28):
        bloque = X[lines:lines+28]
        res.append(bloque)
        
    return np.array(res)


def cargaValores(y):
    res = np.array([int(y[i][j]) for i in np.arange(len(y)) for j in np.arange(len(y[0]))])
    return res



lectura_Xtrain = lecturaFichero("trainingimages")
lectura_Xval = lecturaFichero("validationimages")
lectura_Xtest = lecturaFichero("testimages")

lectura_ytrain = lecturaFichero("traininglabels")
lectura_yval = lecturaFichero("validationlabels")
lectura_ytest = lecturaFichero("testlabels")


X_train_images = cargaImagenes(lectura_Xtrain)
X_val_images = cargaImagenes(lectura_Xval)
X_test_images = cargaImagenes(lectura_Xtest)
    
y_train_images = cargaValores(lectura_ytrain)
y_val_images = cargaValores(lectura_yval)
y_test_images = cargaValores(lectura_ytest)














# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 


#1,2)
# Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
# Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
# Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
# Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
# normst_cancer=NormalizadorStandard()
# normst_cancer.ajusta(Xe_cancer)
# Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)



#7)
# Xc=np.array([["a",1,"c","x"],
#                     ["b",2,"c","y"],
#                     ["c",1,"d","x"],
#                     ["a",2,"d","z"],
#                     ["c",1,"e","y"],
#                     ["c",2,"f","y"]])
# codifica_one_hot(Xc)












