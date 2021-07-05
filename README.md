# PreProcesamiento Dataset

Filtrar datos:
Primero hay que guardar las capturas pcap en una carpeta llamada "capturas_pcap"
>> bash script_filtrarV2.sh

FALTA Revisar filtro: filtra demasiado, limitar a quitar puertos bien conocidos y señalización TCP.
Salida carpeta llamada "capturas_pcap_filtradas"

Convertir de captura de tráfico a archivo binario por paquete:
>> bash script_pkt_to_txt.sh

(usa el ejecutable "analizador" obtenido de compilar analizador.c)
Salida: para cada captura pcap una carpeta con su nombre y dentro subcarpetas por flujos que contienen los paquetes de tráfico binarios.
OJO en algún caso puede que se haga referencia a archivos .txt en vez de .bin

Convertir paquete a imagen:
>> bash script_bin2img.sh capturas_pcap_filtradas/

(usa el python binary2image.py)
Salida: genera dos subcarpetas para cada flujo que contienen las imágenes en 1D y 2D

Si se desea elimnar los archivos binarios (no los procesa la Red neuronal) se puede usar script_filtrar.sh

# Modelo CNN LSTM
modelo_CNN_LSTM.py
Usa TF para GPUs

Genera datos de entrenamiento y test según las carpetas de flujos
Los labels los toma de "archivo_labels.txt" que contiene el nombre de las capturas de tráfico y label asignado.

El modelo propuesto son dos capas CNN luego 2 capas LSTM y después dos capas Densas con Dropouts.

Resultados en mediadas de rendimiento accuracy y loss + Confusion Matrix + Classification Report para comparar resultados por clases.
Por último hay medidas de rendimiento de FPS y Latencia por imagen para comparar con DPU.

# Clasificador_DPU
FALTA documentar


Entrenar modelo:
Sirve el anterior o se puede hacer desde VitisAi
Quantizar modelo:
VitisAi quantize

Compilar modelo:
VitisAi compile

Ejecutar modelo en DPU:

Subir archivo .xmodel, subir carpeta comprimida con imágenes de test y descomprimirla.
Subir archivo words.txt (que contiene las clases) y añadirlo a la carpeta de imágenes de test.
Opcional: para comprobar la precisión del modelo subir archivo labels_test.txt

En dpu_cnn_tfm.ipynb:

(1.Prepare the overlay) Cargar DPU ,modelo y librerías.
(2 Utility functions) Cargar función calculate_softmax, predict_label y sorted_alphanumeric. Las demás funciones no se utilizan.
(3 Use VART) Cargar función run(), medir rendimiento y latencia con time.time()
  Si no se va a comprobar el accuracy se puden mejorar los FPS comentando la línea de file.write()
  





