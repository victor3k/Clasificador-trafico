cd capturas_pcap_filtradas
rm -R capturas_bin
mkdir capturas_bin
for i in `ls`
do
dirname=${i%%.*} # quitar el .pcap
mkdir capturas_bin/$dirname
mv $i capturas_bin/$dirname

cd capturas_bin/$dirname
./../../../analizador -f $i #&>/dev/null
# mv capturas_pcap_filtradas/*.bin capturas_pcap_filtradas/capturas_bin
cd ..
cd ..
mv capturas_bin/$dirname/$i .

echo $i analizado y movido a carpeta $dirname
done
cd ..
rm -d capturas_pcap_filtradas/capturas_bin/capturas_bin
