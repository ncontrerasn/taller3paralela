# Taller 2 Parelela

## Para probar individualmente:

Compilar con: g++ taller2.cpp -o t2 -fopenmp \`pkg-config --cflags --libs opencv\` 

Correr con: ./t2 4k.jpg 4ksobel.jpg 4

## Para correr todo con el script:

Dar permiso al script con: chmod 755 script_ejecutar_todo.sh 

Ejecutarlo con: ./script_ejecutar_todo.sh

**Nota:** para que el script siga con la ejecuciòn de otro caso, hay que cerrar las 2 ventanas que muestran las imàgenes
