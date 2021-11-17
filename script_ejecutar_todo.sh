#!/bin/bash

g++ taller2.cpp -o t2 `pkg-config --cflags --libs opencv`

./t2 720p.jpg 720psobel.jpg 2
./t2 720p.jpg 720psobel.jpg 4
./t2 720p.jpg 720psobel.jpg 8
./t2 720p.jpg 720psobel.jpg 16

./t2 1080p.jpg 1080psobel.jpg 2
./t2 1080p.jpg 1080psobel.jpg 4
./t2 1080p.jpg 1080psobel.jpg 8
./t2 1080p.jpg 1080psobel.jpg 16

./t2 4k.jpg 4ksobel.jpg 2
./t2 4k.jpg 4ksobel.jpg 4
./t2 4k.jpg 4ksobel.jpg 8
./t2 4k.jpg 4ksobel.jpg 16
