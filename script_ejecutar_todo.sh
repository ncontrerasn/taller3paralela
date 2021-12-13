#!/bin/bash

nvcc taller3.cu -o t3 -g `pkg-config --cflags --libs opencv`

./t3 720p.jpg 720psobel.jpg 1 16
./t3 720p.jpg 720psobel.jpg 1 17
./t3 720p.jpg 720psobel.jpg 1 18
./t3 720p.jpg 720psobel.jpg 1 19
./t3 720p.jpg 720psobel.jpg 1 20
./t3 720p.jpg 720psobel.jpg 2 32
./t3 720p.jpg 720psobel.jpg 2 64
./t3 720p.jpg 720psobel.jpg 2 128


./t3 1080p.jpg 1080psobel.jpg 1 32
./t3 1080p.jpg 1080psobel.jpg 1 64
./t3 1080p.jpg 1080psobel.jpg 1 128
./t3 1080p.jpg 1080psobel.jpg 2 16
./t3 1080p.jpg 1080psobel.jpg 2 32
./t3 1080p.jpg 1080psobel.jpg 2 64
./t3 1080p.jpg 1080psobel.jpg 2 128
./t3 1080p.jpg 1080psobel.jpg 3 64
./t3 1080p.jpg 1080psobel.jpg 3 128
./t3 1080p.jpg 1080psobel.jpg 4 64
./t3 1080p.jpg 1080psobel.jpg 4 128


./t3 4k.jpg 4ksobel.jpg 1 128
./t3 4k.jpg 4ksobel.jpg 2 128
./t3 4k.jpg 4ksobel.jpg 3 128
./t3 4k.jpg 4ksobel.jpg 4 128
./t3 4k.jpg 4ksobel.jpg 5 128
./t3 4k.jpg 4ksobel.jpg 6 128
./t3 4k.jpg 4ksobel.jpg 7 128
./t3 4k.jpg 4ksobel.jpg 8 128
./t3 4k.jpg 4ksobel.jpg 9 128
./t3 4k.jpg 4ksobel.jpg 10 128
./t3 4k.jpg 4ksobel.jpg 12 128
./t3 4k.jpg 4ksobel.jpg 256 512
