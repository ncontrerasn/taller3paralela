#!/bin/bash

nvcc taller3.cu -o t3 -fopenmp `pkg-config --cflags --libs opencv` 

./t3 720p.jpg 720psobel.jpg 1 1
./t3 720p.jpg 720psobel.jpg 1 1
./t3 720p.jpg 720psobel.jpg 1 1
./t3 720p.jpg 720psobel.jpg 1 1

./t3 1080p.jpg 1080psobel.jpg 1 1
./t3 1080p.jpg 1080psobel.jpg 1 1
./t3 1080p.jpg 1080psobel.jpg 1 1
./t3 1080p.jpg 1080psobel.jpg 1 1

./t3 4k.jpg 4ksobel.jpg 1 1
./t3 4k.jpg 4ksobel.jpg 1 1
./t3 4k.jpg 4ksobel.jpg 1 1
./t3 4k.jpg 4ksobel.jpg 1 1
