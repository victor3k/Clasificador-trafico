mkdir capturas_pcap_filtradas
for i in `ls capturas_pcap/`
do
tshark -r capturas_pcap/$i -Y "tcp && tcp.len != 0 || ssl || gquic || data " -w capturas_pcap_filtradas/f_$i
echo "$i filtrado y copiado"
done
