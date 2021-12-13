# Taller 3 Parelela

## Para probar individualmente:

Compilar con: nvcc taller3.cu -o taller3 -g \`pkg-config --cflags --libs opencv\`

Correr con: ./t3 4k.jpg 4ksobel.jpg 2 6

**Nota:** los parámetros son: nombre de la imagen de entrada, nombre de la imagen de de salida, número de bloques y número de hilos por bloque.

## Para correr todo con el script:

Dar permiso al script con: chmod 755 script_ejecutar_todo.sh 

Ejecutar con: ./script_ejecutar_todo.sh
