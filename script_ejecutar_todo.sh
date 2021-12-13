#!/bin/bash

nvcc taller3.cu -o taller3 -g `pkg-config --cflags --libs opencv`

./t3 720p.jpg 720psobel.jpg 1 16
./t3 720p.jpg 720psobel.jpg 1 64
./t3 720p.jpg 720psobel.jpg 1 128
./t3 720p.jpg 720psobel.jpg 12 64
./t3 720p.jpg 720psobel.jpg 12 128
./t3 720p.jpg 720psobel.jpg 24 128
./t3 720p.jpg 720psobel.jpg 48 128
./t3 720p.jpg 720psobel.jpg 24 256
./t3 720p.jpg 720psobel.jpg 96 128
./t3 720p.jpg 720psobel.jpg 24 512
./t3 720p.jpg 720psobel.jpg 96 512
./t3 720p.jpg 720psobel.jpg 240 1280

./t3 1080p.jpg 1080psobel.jpg 1 16
./t3 1080p.jpg 1080psobel.jpg 1 64
./t3 1080p.jpg 1080psobel.jpg 1 128
./t3 1080p.jpg 1080psobel.jpg 12 64
./t3 1080p.jpg 1080psobel.jpg 12 128
./t3 1080p.jpg 1080psobel.jpg 24 128
./t3 1080p.jpg 1080psobel.jpg 48 128
./t3 1080p.jpg 1080psobel.jpg 24 256
./t3 1080p.jpg 1080psobel.jpg 96 128
./t3 1080p.jpg 1080psobel.jpg 24 512
./t3 1080p.jpg 1080psobel.jpg 96 512
./t3 1080p.jpg 1080psobel.jpg 240 1280

./t3 4k.jpg 4ksobel.jpg 1 16
./t3 4k.jpg 4ksobel.jpg 1 64
./t3 4k.jpg 4ksobel.jpg 1 128
./t3 4k.jpg 4ksobel.jpg 12 64
./t3 4k.jpg 4ksobel.jpg 12 128
./t3 4k.jpg 4ksobel.jpg 24 128
./t3 4k.jpg 4ksobel.jpg 48 128
./t3 4k.jpg 4ksobel.jpg 24 256
./t3 4k.jpg 4ksobel.jpg 96 128
./t3 4k.jpg 4ksobel.jpg 24 512
./t3 4k.jpg 4ksobel.jpg 96 512
./t3 4k.jpg 4ksobel.jpg 240 1280
