import os, math
import argparse
from PIL import Image
from queue import Queue
from threading import Thread


def getBinaryData(filename):

	binary_values = []

	with open(filename, 'rb') as fileobject:

		data = fileobject.read(1)

		while data != b'':
			binary_values.append(ord(data))
			data = fileobject.read(1)

	return binary_values


def createGreyScaleImage(filename, width=None):

	greyscale_data  = getBinaryData(filename)
	size            = get_size(len(greyscale_data), width)
	carpeta_1d = '1Dimension'
	save_file(filename, greyscale_data, size, 'L',carpeta_1d)
	size            = get_size_2d(len(greyscale_data), width)
	carpeta_2d = '2Dimension'
	save_file(filename, greyscale_data, size, 'L',carpeta_2d)


def save_file(filename, data, size, image_type,carpeta):

	try:
		image = Image.new(image_type, size)
		image.putdata(data)
		 
		dirname     = os.path.dirname(filename)
		name, _     = os.path.splitext(filename)
		name        = os.path.basename(name)
		imagename   = dirname + os.sep + carpeta+ os.sep + name + '_'+image_type+ '.png'
		os.makedirs(os.path.dirname(imagename), exist_ok=True)

		image.save(imagename)
		print('The file', imagename, 'saved.')
	except Exception as err:
		print(err)


def get_size(data_length, width=None):
	
	#Definimos la anchura de nuestra imagen 
	width = 1024
	height = 1 #Conseguimos asi tener una imagen en 1D

	return (width, height)

def get_size_2d(data_length, width=None):

	# Definimos la anchura y altura de nuestra imagen 2D
	return (32, 32)


def run(file_queue, width):

	while not file_queue.empty():
		filename = file_queue.get()
		createGreyScaleImage(filename, width)
		file_queue.task_done()


def main(input_dir, width=None, thread_number=7):

	# Obtenemos todos los ficheros del directorio y lo añadimos a la cola
	file_queue = Queue()
	for root, directories, files in os.walk(input_dir):
		for filename in files:
			file_path = os.path.join(root, filename)
			file_queue.put(file_path)

	
	for index in range(thread_number):
		thread = Thread(target=run, args=(file_queue, width))
		thread.daemon = True
		thread.start()
	file_queue.join()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='binar2image.py', description="Convertidor de archivos binarios en imagenes ")
	parser.add_argument(dest='input_dir', help='Introducir un directorio donde se encuentran los archivos')

	args = parser.parse_args()

	main(args.input_dir, width=None)
