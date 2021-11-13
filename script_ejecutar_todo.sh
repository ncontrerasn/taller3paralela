#!/bin/bash

g++ taller1.cpp -o t1 `pkg-config --cflags --libs opencv`

./t1 720p.jpg 720psobel.jpg

./t1 1080p.jpg 1080psobel.jpg

./t1 4k.jpg 4ksobel.jpg