echo empezando a convertir texto a imagenes...
cd capturas_pcap_filtradas/capturas_txt
for i in `ls`
do
	cd $i

	for j in `ls`
	do

		python3 ./../../../binary2image.py $j

	done

	echo paquetes $i convertidos a imagenes 1D y 2D

	cd ..

done
cd ..
cd ..
echo paquetes convertidos a imagenes 1D y 2D