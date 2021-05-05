# Clasificador_DPU

Entrenar modelo:

Quantizar modelo:

Compilar modelo:

Ejecutar modelo en DPU:

Subir archivo .xmodel, subir carpeta comprimida con imágenes de test y descomprimirla.
Subir archivo words.txt (que contiene las clases) y añadirlo a la carpeta de imágenes de test.
Opcional: para comprobar la precisión del modelo subir archivo labels_test.txt

En dpu_cnn_tfm.ipynb:

(1.Prepare the overlay) Cargar DPU ,modelo y librerías.
(2 Utility functions) Cargar función calculate_softmax, predict_label y sorted_alphanumeric. Las demás funciones no se utilizan.
(3 Use VART) Cargar función run(), medir rendimiento y latencia con time.time()
  Si no se va a comprobar el accuracy se puden mejorar los FPS comentando la línea de file.write()
  





