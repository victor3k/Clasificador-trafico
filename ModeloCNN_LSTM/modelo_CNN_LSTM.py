#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:02:29 2020

@author: victor
Referencia 1 = https://github.com/python-engineer/tensorflow-course/blob/master/05_cnn.py
Referencia 2 = Apuntes PASM de UAM EPS (Aytami)
Referencia 3 = TFM Ignacio Sotomonte
"""

# Función para ordenar las listas de archivos como en windows
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


import os, shutil, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Flatten, MaxPool2D, Dropout, TimeDistributed, LSTM


import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Directory where dataset is stored
# cuantos pixeles tienen las imagenes 2D (si es 1D sera px^2)
pixeles = '32' # 1D: 28 , 32 , 38 o 2D: 784, 1024 , 1444
pixeles_int = int(pixeles)
# Número de dimensiones
dir_dimension = '2Dimension';

base_dir = 'D:/Escritorio/TFM/capturas_pcap_filtradas_32px_4kpkts/capturas_txt'

# Obtener los labels de cada fichero
archivo_nombre_labels = 'D:/Escritorio/TFM/archivos_labels.txt';
datos_labels = []
with open(archivo_nombre_labels) as f:
    for linea in f.readlines():
        datos_labels.append(linea.strip())

lista_datos = []
for i in range(len(datos_labels)):
    lista_datos.append(datos_labels[i].split())

nombre_columnas = ['fichero', 'label']

dicc_datos = []
for j in range(len(lista_datos)):
    dicc_datos.append(dict(zip(nombre_columnas,lista_datos[j])))



contador = 0

# Cuantos paquetes se usan del dataset. Más mejor pero ojo con descompensar clases como Email
n_pcap = 120000

# obtener datos train y test
size_train = 7
size_test = 10 - size_train

train_images_list = []
test_images_list = []
train_labels_list = []
test_labels_list = []

# Version nueva -> train test segun flujos
names_labels = [x[1] for x in lista_datos]
class_names = np.unique(names_labels)
nClasses = len(class_names)

# limite de paquetes train y test
max_pkts_train = round(n_pcap/nClasses*size_train/10)
max_pkts_test = round(n_pcap/nClasses*size_test/10)

# Genero variable de contador de paquetes
contador_pkts_train = np.zeros(nClasses)
contador_pkts_test = np.zeros(nClasses)

dirs = sorted_alphanumeric(os.listdir(base_dir))

#%%
# Proceso de generar las imágenes guardadas por flujos. Se puede saltar este paso usando np.load
list_test_dirs = []

# Para cada captura pcap se usan algunos flujos de train y otros distintos de test. El label de cada imagen viene dado por el nombre de la captura
for fichero in dirs:
    print(fichero)
    dir_flujo = os.path.join(base_dir,fichero)
    dirs_flujo = sorted_alphanumeric(os.listdir(dir_flujo))
    
    for k in range(len(dicc_datos)):
        if(dicc_datos[k]["fichero"] == fichero):
            label = dicc_datos[k]["label"]
            break
        else:
            label = "no_label_found"
    print(label)
        
    # para asegurar que hay datos de todos los labels se establece un contador por label
    indice_cl = 0
    for x in class_names:
        if x == label:
            break
        else:
            indice_cl = indice_cl + 1
            
    print(contador_pkts_train[indice_cl])
    print(contador_pkts_test[indice_cl])
    
    # cuantos paquetes queremos ver (ojo es aproximado ya q toma el fichero entero)
    if contador_pkts_train[indice_cl] < max_pkts_train or contador_pkts_test[indice_cl] < max_pkts_test:
        
        num_flujos = len(dirs_flujo)
        #print("Num flujos")
        #print(num_flujos)
        num_flujos_train = round(num_flujos*size_train/10)
        num_flujos_test = round(num_flujos*size_test/10)
        contador_flujos = 0

        for flujo in dirs_flujo:
            #print(flujo)
            dir_imagen = os.path.join(dir_flujo,flujo,dir_dimension)
            dirs_imagen = sorted_alphanumeric(os.listdir(dir_imagen))

            #print(contador_flujos)
            contador_flujos = contador_flujos + 1
            #print(flujo)
            #print(contador_flujos)

            for file in dirs_imagen:
                #print(file)
                dir_file = os.path.join(dir_imagen,file)
                im = plt.imread(dir_file)
				
				# Para coger 7 flujos train luego 3 flujos de test en vez de los primeros de train y los últimos de test.
                if(contador_flujos%10 < size_train): 
                    #print("se guarda en train")
                    if(contador_pkts_train[indice_cl] < max_pkts_train):
                        train_images_list.append(im)		                        
                        train_labels_list.append(label)
                        contador_pkts_train[indice_cl] = contador_pkts_train[indice_cl] + 1
                        
                else:
                    #print("se guarda en test")
                    if(contador_pkts_test[indice_cl] < max_pkts_test):
                        test_images_list.append(im)
                        test_labels_list.append(label)
                        contador_pkts_test[indice_cl] = contador_pkts_test[indice_cl] + 1
                        # Guardar lista de imagenes de test
                        list_test_dirs.append(dir_file)
                                            

    print(contador_pkts_train[indice_cl])
    print(contador_pkts_test[indice_cl])
                    

# convierto a tipo array de float64 para la cnn
train_images = np.asarray(train_images_list, dtype=np.float64)
test_images = np.asarray(test_images_list, dtype=np.float64)

# convertir de list de strings a array uint8, usar class_names
train_labels_array = np.zeros(len(train_labels_list))
test_labels_array = np.zeros(len(test_labels_list))

for i in range(nClasses):
    for j in range(len(train_labels_list)):
        if train_labels_list[j] == class_names[i]:
            train_labels_array[j] = i

for i in range(nClasses):
    for j in range(len(test_labels_list)):
        if test_labels_list[j] == class_names[i]:
            test_labels_array[j] = i    

train_labels = np.asarray(train_labels_array, dtype = np.uint8)
test_labels = np.asarray(test_labels_array, dtype = np.uint8)

# Si solo tiene una dimension la imagen será 784 o 1024 o 1444 si no serán 2D 28x28, 32x32 o 38x38
if dir_dimension == '1Dimension':
    train_images = train_images.reshape(len(train_images),pixeles_int,1) # 28,28,1
    test_images = test_images.reshape(len(test_images),pixeles_int,1) # 28,28,1
    train_labels = train_labels.reshape(len(train_labels),1)
    test_labels = test_labels.reshape(len(test_labels),1)
else:
    train_images = train_images.reshape(len(train_images),pixeles_int,pixeles_int,1) # 28,28,1
    test_images = test_images.reshape(len(test_images),pixeles_int,pixeles_int,1) # 28,28,1
    train_labels = train_labels.reshape(len(train_labels),1)
    test_labels = test_labels.reshape(len(test_labels),1)
    

#%%
# Libreria para Guardar y Cargar sesiones
import dill
        
filename1 = 'D:/Escritorio/TFM/prueba_soloCNN_1444px_5clases.pkl'
#dill.dump_session(filename1)
#dill.load_session(filename1)

# Guardar datos como npy
np.save('D:/Escritorio/TFM/test_labels.npy',test_labels)
np.save('D:/Escritorio/TFM/test_images.npy',test_images)
np.save('D:/Escritorio/TFM/train_labels.npy',train_labels)
np.save('D:/Escritorio/TFM/train_images.npy',train_images)

# Usar np.load para cargar las imágenes

# Model aqui va la cnn y lstm en 1d o 2d
if dir_dimension == '1Dimension':
    
    model = tensorflow.keras.models.Sequential([
    InputLayer(input_shape=(pixeles_int, 1), name='input_data'),
    Conv1D(32, 3, activation='relu'),
    MaxPool1D(pool_size=(2)),
    Conv1D(64, 3, activation='relu'),
    MaxPool1D(pool_size=(2)),
    TimeDistributed(Flatten()),
    LSTM(50,return_sequences=True),
    Dropout(0.2),
    Dense(64, activation='relu'),
    LSTM(25,return_sequences=True),
    Dropout(0.2),
    Flatten(),
    Dropout(0.2),
    Dense(5, activation='softmax', name='output_logits_tfm')
])
    
else:
    model = tensorflow.keras.models.Sequential([
    InputLayer(input_shape=(pixeles_int, pixeles_int, 1), name='input_data'),
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    TimeDistributed(Flatten()),
    LSTM(50,return_sequences=True),
    Dropout(0.2),
    Dense(64, activation='relu'),
    LSTM(25,return_sequences=True),
    Dropout(0.2),
    Flatten(),
    Dropout(0.2),
    Dense(5, activation='softmax', name='output_logits_tfm')
])
   

print(model.summary())
#import sys; sys.exit()

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["sparse_categorical_accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

#%%
# Entrenamiento del modelo.

batch_size = 64
epochs = 50

# train_image y test_image tienen que ser array of float64
# train_label y test_label tienen que ser array of uint8

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2,validation_data=(test_images,test_labels))

#%%
# Guardamos sesion
filename2 = 'D:/Escritorio/TFM/prueba(LSTM_desmodel).pkl'
#dill.dump_session(filename2)

#%%
# Guardamos la red, por si la utilizazmos despues

model.save('D:/Escritorio/TFM/CNN_2D.h5')
modelo = keras.models.load_model('D:/Escritorio/TFM/CNN_2D.h5')

# evaluation
test_loss, test_acc = modelo.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

n_epochs = range(1,len(acc)+1)

plt.plot(n_epochs,loss,'bo',label='Training loss')
plt.plot(n_epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(n_epochs,acc,'bo',label='Training accuracy')
plt.plot(n_epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% 
# Predict, Clasficication Report y Confusion Matrix

prediction = model.predict(test_images, batch_size=batch_size, verbose=1)

from sklearn.metrics import classification_report, confusion_matrix

prediction_true_false = prediction > 0.5

prediction_1d = np.zeros(len(prediction))
for i in range(len(prediction)):
    indice = np.argmax(prediction[i])
    prediction_1d[i] = indice
    
test_labels_matrix = np.zeros((len(test_labels), nClasses))
for i in range(len(test_labels)):
    test_labels_matrix[i][test_labels[i][0]] = 1
        
con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=prediction_1d)
print('Confusion Matrix: ',con_mat)
data = con_mat

target_names = ["Class {}".format(i) for i in range(nClasses)]

classification_report = classification_report(test_labels,prediction_1d, target_names = class_names) # cambiar class_names por target_names
print('Classification Report: \n',classification_report)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


flag_con_mat = 0

if flag_con_mat == 1:
    # crear array = con_mat a mano

    df_cm = pd.DataFrame(array, class_names, class_names)
    sn.heatmap(df_cm,annot_kws={"size": 16}) # font size
    


# %%
# Medidas de rendimiento accuracy 
# througput total
# latencia media por imagen

# si flag 0 no ejecuta si flag 1 sí ejecuta
flag_run_rendimiento = 0

import time

if flag_run_rendimiento == 1:
    batch = 64
    image_list = []
    
    time1_2 = time.time()
    for i in list_test_dirs:
        im = plt.imread(i)
        image_list.append(im)
    
    image_list_np = np.asarray(image_list, dtype=np.float64)
    image_list_np_rs = image_list_np.reshape(len(list_test_dirs),pixeles_int,pixeles_int,1)
        
    predict = modelo.predict(image_list_np_rs,batch_size=batch,verbose=0)
    
    time2_2 = time.time()
    fps_2 = len(list_test_dirs)/(time2_2-time1_2)
    print('Con batch 64')
    print('FPS: ',fps_2)
    print('Tiempo total: ',time2_2-time1_2)
    
    # solo un batch para calcular latencia
    time3_2 = time.time()
    
    image_list = []
    
    for i in list_test_dirs[0:batch]:
        im = plt.imread(i)
        image_list.append(im)
    
    image_list_np = np.asarray(image_list, dtype=np.float64)
    image_list_np_rs = image_list_np.reshape(batch,pixeles_int,pixeles_int,1)
    
    time4_2 = time.time()
    
    predict = modelo.predict(image_list_np_rs,batch_size=batch,verbose=0)
    
    time5_2 = time.time()
        
    print('Latency batch: ',time5_2 - time3_2)
    print('Latency carga img batch: ',time4_2 - time3_2)
    print('Latency predict: ',time5_2 - time3_2)
else:
    print("flag_run_rendimiento = 0 No se ejecuta")


