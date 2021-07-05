/***************************************************************************
 analizador.c

 main y analizar_paquete()

 Compila: gcc -Wall -o analizador analizador.c -lpcap
 Ejecuta: ./analizador -f fichero.pcap
 		  ./analizador -e conexion      ----    NO ES NECESARIO PARA EL TFM

 Autor: Victor Morales Gomez

 Toma como codigo de partida el codigo de TFG del autor y el codigo de TFM de Ignacio Sotomonte.
 
 Y pagina web de filtrado tcp/ip SNIFFEX-1.C

 Cambios:	
 El filtrado de paquetes se escribe directamente desde el terminal.
 Por ejemplo.
 ./disector -f quic.pcap -e "src or dst 192.168.0.1"

 Usaria como filtro src or dst 192.168.0.1 y filtraria los paquetes que tengan
 192.168.0.1 en ip dst o ip src.

 Para la estructura de datos se utiliza una tabla hash.
 El código de partida de esta tabla hash se puede consultar aqui:

 https://www.tutorialspoint.com/data_structures_algorithms/hash_table_program_in_c.htm

***************************************************************************/

#include "analizador.h"
//#include "hashtable.h"
#include <endian.h>
#include <time.h>
#include <limits.h>

#define SIZE 20000

int numpktout = 10000;
int quic_port = 443;


char *token; // nombre del fichero
char name[50];

int contador_flujos = 0;

typedef struct Flujo {
  int id_f;
}Flujo;

struct Flujo* hashArrayFlujo[SIZE];

int main(int argc, char **argv)
{
	char errbuf[PCAP_ERRBUF_SIZE];
	
	int flag_e = 0;
	int flag_n = 0;
	int long_index = 0, retorno = 0;
	char opt;
	
	/* DECLARACION DE VARIABLES DE FILTRO */
	/* VER SNIFFEX-1.C */

	char filter_exp[] = "";		/* filter expression*/
	struct bpf_program fp;			/* compiled filter program (expression) */
	//bpf_u_int32 mask = 0;			/* subnet mask */
	bpf_u_int32 net = 0;			/* ip */

	// Nombre del fichero a global
    strcpy(name,argv[2] );
	const char s[2] = ".";
	// char * prueba;
	// prueba = strrchr(name,'/');
	// printf("Valor despues / = %s\n",prueba );
	token = strtok(name, s);
	//printf("token=%s\n",token );


	if (signal(SIGINT, handleSignal) == SIG_ERR) {
		//printf("Error: Fallo al capturar la senal SIGINT.\n");
		exit(ERROR);
	}

	if (argc == 1) {
		exit(ERROR);
	}

	static struct option options[] = {
		{"f", required_argument, 0, 'f'},
		{"i",required_argument, 0,'i'},
		{"e", required_argument, 0, 'e'},
		{"n",required_argument,0,'n'},
		{"p",required_argument,0,'p'},
		{"h", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long_only(argc, argv, "f:i:e:n:p:h", options, &long_index)) != -1) {
		switch (opt) {
		case 'i' :
			if(descr) { // comprobamos que no se ha abierto ninguna otra interfaz o fichero
				//printf("Ha seleccionado más de una fuente de datos\n");
				pcap_close(descr);
				exit(ERROR);
			}
		
			if ( (descr = pcap_open_live(optarg, 1518, 0, 100, errbuf)) == NULL){
				//printf("Error: pcap_open_live(): Interface: %s, %s %s %d.\n", optarg,errbuf,__FILE__,__LINE__);
				exit(ERROR);
			}
			break;

		case 'f' :
			if(descr) { // comprobamos que no se ha abierto ninguna otra interfaz o fichero
				//printf("Ha seleccionado más de una fuente de datos\n");
				pcap_close(descr);
				exit(ERROR);
			}

			if ((descr = pcap_open_offline(optarg, errbuf)) == NULL) {
				//printf("Error: pcap_open_offline(): File: %s, %s %s %d.\n", optarg, errbuf, __FILE__, __LINE__);
				exit(ERROR);
			}

			break;

		case 'e' :
			//printf("Filtro introducido:%s \n",argv[4]);
			flag_e = 1;						
			break;

		case 'h' :
			printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
			exit(ERROR);
			break;

		case 'n' :
			if(flag_e == 0){
				printf("Please enter a filter before variables:\n");
				printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
				exit(ERROR);
			}
			//printf("Variables introducidas: num pkts delete= %s, quic port= %s\n",argv[6],argv[7]);
			numpktout = atoi(argv[6]);
			flag_n = 1;

			//printf("Num pkt %d\n",numpktout );
			//printf("flag_n %d\n",flag_n );
			//printf("quic port %d\n",quic_port );
			break;

		case 'p' :
			if(flag_e == 0){
				printf("Please enter a filter before variables:\n");
				printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
				exit(ERROR);
			}
			//printf("Variables introducidas: num pkts delete= %s, quic port= %s\n",argv[6],argv[7]);
			if(flag_n == 0)
				quic_port = atoi(argv[6]);
			else
				quic_port = atoi(argv[8]);


			//printf("Num pkt %d\n",numpktout );
			//printf("quic port %d\n",quic_port );
			break;

		case '?' :
		default:
			//printf("Error. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp'']: %d\n", argv[0], argc);
			exit(ERROR);
			break;
		}
	}

	if (!descr) {
		//printf("No selecciono ningún origen de paquetes.\n");
		return ERROR;
	}

	//printf("\n");

	if(argc == 5 && flag_e == 1){
		//printf("Se ha aplicado el filtro anterior.\n");
		if (pcap_compile(descr, &fp, argv[4], 0, net) == -1) {
		fprintf(stderr, "Couldn't parse filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);
		}
	}	
	else{
		//printf("No se aplica filtro.\n");
		if (pcap_compile(descr, &fp, filter_exp, 0, net) == -1) {
		fprintf(stderr, "Couldn't parse filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);	
		}
	}

	/* apply the compiled filter */
	if (pcap_setfilter(descr, &fp) == -1) {
		fprintf(stderr, "Couldn't install filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);
	}

	/* Precarga de memoria*/
	
	//load_mem();
	int i = 0;
	//printf("prueba load mem\n");

	struct Flujo *item = (struct Flujo*) malloc(SIZE*sizeof(struct Flujo));

	for (i = 0; i < SIZE; i++){
		item->id_f = 0;
		hashArrayFlujo[i] = item;
		item ++;
	}	
		//printf("fin prueba load mem\n");

	retorno=pcap_loop(descr,NO_LIMIT,analizar_paquete,NULL);
	switch(retorno)	{
		case OK:
			//printf("Traza leída\n");
			break;
		case PACK_ERR: 
			//printf("Error leyendo paquetes\n");
			break;
		case BREAKLOOP: 
			//printf("pcap_breakloop llamado\n");
			break;
	}
	//printf("Se procesaron %"PRIu64" paquetes.\n\n", contador);
	pcap_close(descr);

	return OK;
}

void analizar_paquete(u_char *user,const struct pcap_pkthdr *hdr, const uint8_t *pack)
{
	(void)user;
	//printf("*******************************************************\n");
	//printf("-------------------------------------------------------\n");
	//printf("Nuevo paquete capturado el %s\n", ctime((const time_t *) & (hdr->ts.tv_sec)));
	
	contador++;

	int flag_cabecera = 1; // si esta a 0 se añade la cabecera al txt si esta a 1 solo hay payload

	int hash_flujo = 0;
	int id_flujo = 0;

	// Variables para bucles
	int i = 0;
	int j = 0;

	// Variables para cond logicas
	int offset = 0;		// tag para ver si tiene paquete ip 
	int Udp = 0;
	int Tcp = 0;	// Da warning porque esta comentada la parte de analizar tcp

	// Variables para guardar longitud de paquete
	u_int ip_size = 0;
	u_int udp_size = 8;

	// Variables para guardar en tabla
	//char hostname[100] = {"NULL"};
	int key = 0;
	double timesec = 0;
	double timeus = 0;
	double time = 0;

	int ipproto = 0;

	int src_addr[4] = {0,0,0,0};
	int dst_addr[4] = {0,0,0,0};
	int src_port = 0;
	int dst_port = 0;
	//uint64_t cid = 0;
	//int version = 0;
	int ip_len = 0;
	int ip_header_len = 0;
	int udp_len = 0;
	int udp_header_len = 0;

	int size_tcp = 0;

	timesec = hdr->ts.tv_sec;
	timeus = hdr->ts.tv_usec;
	time = timesec + timeus*0.000001;

	/*Para campos ETH se usa casting ya que siempre siguen el mismo orden */
	const struct sniff_ethernet *ethernet;
	ethernet = (struct sniff_ethernet*)(pack);

	// Hay que comprobar que el paquete sea IP, si no se descarta.

	// if(ethernet->ether_type != 8){
	// 	//printf("No es un paquete IP, no se analiza");
	// 	//printf("\n");
	// 	return;
	// }
	// else{
		//printf("-------------------------------------------------------\n");
		//printf("Es un paquete IP, se analiza:\n");

	/*Para campos IP se usa casting ya que siempre siguen el mismo orden */
	const struct sniff_ip *ip;
	ip = (struct sniff_ip*)(pack + ETH_HLEN);
/*
	//printf("Version IP= ");
	//printf("%u", (((ip)->ip_vhl >> 4) & 0x0f));
	//printf("\n");
	//printf("IP Longitud de Cabecera= ");
*/		
	ip_size = 4*(ip->ip_vhl&0xf);
	ip_header_len = ip_size;
	////printf("ip header size%d\n",ip_size );
/*		//printf("%u Bytes", ip_size);
	//printf("\n");

	//printf("IP Longitud Total= ");
	//printf("%u Bytes", ntohs(ip->ip_len));
*/
	ip_len = ntohs(ip->ip_len);
	////printf("ip len: %d\n",ip_len );
/*
	//printf("\n");

	//printf("Posicion= ");
*/
	offset = 8*(ntohs((ip->ip_off))&0x1FFF);

	if(ip->ip_p == 17){			
		//printf("Es un paquete UDP\n");
		Udp = 1;
		ipproto = ip->ip_p;
	}
	else if(ip->ip_p == 6){		
		//printf("Es un paquete TCP\n");
		Tcp = 1;
		ipproto = ip->ip_p;
	}
	else{
		//printf("No es un paquete UDP ni TCP, no lo analizamos\n");
		Tcp = 0;
		Udp = 0;
		//return;
	}

	//printf("Direccion IP Origen= ");
	//printf("%u", ip->ip_src[0]);
	src_addr[0] = ip->ip_src[0];
	for (i = 1; i <IP_ALEN; i++) {
		//printf(".%u", ip->ip_src[i]);
		src_addr[i] = ip->ip_src[i];
	}
	//printf("\n");

	//printf("Direccion IP Destino= ");
	//printf("%u", ip->ip_dst[0]);
    dst_addr[0] = ip->ip_dst[0];
	for (j = 1; j <IP_ALEN; j++) {
		//printf(".%u", ip->ip_dst[j]);
    	dst_addr[j] = ip->ip_dst[j];
	}	
	//printf("\n");

	if(Udp == 1 && offset == 0){ 
		//printf("-------------------------------------------------------");
		//printf("\n");

		/*Para campos UDP se usa casting ya que siempre siguen el mismo orden */
		const struct sniff_udp *udp;
		udp = (struct sniff_udp*)(pack + ETH_HLEN + ip_size);

		//printf("Es un paquete UDP, se analiza:");
		//printf("\n");		
		//printf("Puerto Origen= ");		
		//printf("%u", ntohs(udp->udp_sport));
		src_port = ntohs(udp->udp_sport);
		//printf("\n");
	
		//printf("Puerto Destino= ");
		//printf("%u", ntohs(udp->udp_dport));
		dst_port = ntohs(udp->udp_dport);
		//printf("\n");
		
/*
		//printf("Longitud= ");
				
		//printf("%u", ntohs(udp->udp_length));
		//printf("\n");
*/			

		udp_len = ntohs(udp->udp_length);
		////printf("udp len: %d\n",udp_len );
		udp_header_len = udp_size;
		////printf("udp header size %d\n",udp_header_len );


		//printf("-------------------------------------------------------");
		//printf("\n");
	
	} // Cierre de cond udp

	else if(Tcp == 1 && offset == 0){
			
		//printf("-------------------------------------------------------");
		//printf("\n");

		const struct sniff_tcp *tcp;
		tcp = (struct sniff_tcp*)(pack + ETH_HLEN + ip_size);

		//printf("Es un paquete TCP, se analiza:");

		//printf("\n");
		//printf("Puerto Origen= ");

		//printf("%u", ntohs(tcp->th_sport));
		src_port = ntohs(tcp->th_sport);

		//printf("\n");

		//printf("Puerto Destino= ");

		//printf("%u", ntohs(tcp->th_dport));
		dst_port = ntohs(tcp->th_dport);
		//printf("\n");


		size_tcp = TH_OFF(tcp)*4;
		//printf("Longitud de la cabecera tcp: %d \n",size_tcp);
		
	}


	//printf("Quintupla del Flujo = %d %d.%d.%d.%d %d.%d.%d.%d %u %u \n",ipproto,src_addr[0],src_addr[1],src_addr[2],src_addr[3],dst_addr[0],dst_addr[1],dst_addr[2],dst_addr[3],src_port,dst_port);
	key =  src_addr[0]*10000+ src_addr[1]*10000;
	key +=  src_addr[2]*10000+ src_addr[3]*10000;
	key +=  dst_addr[0]+ dst_addr[1];
	key +=  dst_addr[2]+ dst_addr[3];

	key +=  src_port*10000 +  dst_port + ipproto;
	//printf("Key: %d\n",key);

	// Calcular el hash del flujo
	hash_flujo = key % SIZE;

	// Si flujo esta creado recupero su ID si no genero nuevo id

	if (hashArrayFlujo[hash_flujo]->id_f == 0){
		contador_flujos = contador_flujos + 1;
		hashArrayFlujo[hash_flujo]->id_f = contador_flujos;
		id_flujo = hashArrayFlujo[hash_flujo]->id_f;

		char dir[] = {0};
		sprintf(dir, "%s_%d",token, id_flujo);
		//printf("dir: %s\n", dir);

		int result = mkdir(dir,0777);
		//printf("Ha creado dir\n");
	}
	else{
		id_flujo = hashArrayFlujo[hash_flujo]->id_f;
	}

	//printf("id_flujo: %d\n",id_flujo);

	//printf("Guardo archivo: %s_%d_%ld\n",token,id_flujo,contador);

	char buffer[100]={0};

	// printf("%s\n",token );

	sprintf(buffer, "%s_%d/%ld.bin",token, id_flujo, contador);
	//printf("%s\n", buffer);
	FILE *file = fopen(buffer,"w");

	//printf("file creada\n");
	//printf("contador %ld, len %d\n",contador,hdr->len);

	char num = 0;

	// Para quitar las cabeceras ETH,IP,UDP/TCP 
	int ini = ETH_HLEN + ip_size;
	if (ip->ip_p==17){//UDP
		ini = ini + 8;
	}
	if (ip->ip_p==6){//TCP
		ini = ini + size_tcp;
	}

	// según flag cabcera se añade o no la cabcera al bin
	int suma_cabecera = 0;
	if(flag_cabecera == 0){

		suma_cabecera = 0;
	}
	else if(flag_cabecera == 1){

		suma_cabecera = ini;

	}

	int tam_img = 1024;

	for (i = 0 + suma_cabecera; i < tam_img + suma_cabecera; i++){
		if(i < hdr->len){
			//printf("guardar pack[%d] = \n",i);
			fprintf(file,"%c", pack[i]);
		}
		else{
			//printf("guardar %d = 0\n",i);
			fprintf(file,"%c", num);
		}
	}

	//printf("Terminado de guardar archivo: %s_%d_%ld\n",token,id_flujo,contador);

	fclose(file);
}

