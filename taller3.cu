#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/time.h>
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <string>
#include <math.h>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
using std::ofstream;

//Se encarga de realiza el algoritmo de sobel a la imagen
void noBorderProcessing(Mat imgGray, Mat imgSobel, float **Kernel, float **Kernel2)
{
    //Se recorre la imagen completa exclugendo los pixeles borde para que la
    //operacion de convolucion siguiente puede hacerse sin padding

    //#pragma omp parallel for collapse(2)
    for (int y = 1; y < imgGray.rows - 1; y++)
    {
        for (int x = 1; x < imgGray.cols - 1; x++)
        {
            //Se debe realizar la operacion por cada uno de los colores RGB que se encuentran en cada pixel
            for (int f = 0; f < 3; f++)
            {
                float sum = 0.0;
                float sum2 = 0.0;
                float fsum = 0.0;
                //Se establece con estos dos fors la operacion de convolucion entre la matriz de la imagen y los kernels
                for (int k = -1; k <= 1; k++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        sum = sum + Kernel[j + 1][k + 1] * imgGray.at<Vec3b>(y - j, x - k)[f];
                        sum2 = sum2 + Kernel2[j + 1][k + 1] * imgGray.at<Vec3b>(y - j, x - k)[f];
                    }
                }
                //Segun dicta el algoritmo se aplica la siguiente operacion
                fsum = ceil(sqrt((sum * sum) + (sum2 * sum2)));
                //el valor resultante se substituye en el pixel correspondiente de la imagen objetivo
                imgSobel.at<Vec3b>(y, x)[f] = fsum;
            }
        }
    }
}

//aca no es necesario pasarlas de a 3 pff jaja. esto seria para sobel
__global__ void sobel(unsigned char *d_imgGray, unsigned char *d_imgSobel, int cols, int rows, int numberElements, int totalThreads, float **Kernel, float **Kernel2)
{

    int offSet = cols * 3 + 3;
    int YoffSet = cols * 3;
    int x;

    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    int initIteration = ((numberElements / totalThreads) * index) + offSet;
    int endIteration = initIteration + (numberElements / totalThreads) - 1;

    if (endIteration < (numberElements - offSet*5) && index<4)
    {
        printf("Entro %i, inicio: %i y final %i\n", index, initIteration, endIteration);
        for (x = initIteration; x < endIteration; x = x + 3)
        {
            //Se debe realizar la operacion por cada uno de los colores RGB que se encuentran en cada pixel
            for (int f = 0; f < 3; f++)
            {
                float sum = 0.0;
                float sum2 = 0.0;
                float fsum = 0.0;
                //Se establece con estos dos fors la operacion de convolucion entre la matriz de la imagen y los kernels
                for (int k = -1; k <= 1; k++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        printf("value  %u\n", Kernel[j + 1][k + 1]);
                        sum = sum + Kernel[j + 1][k + 1] * d_imgGray[x + YoffSet * j + k*3 + f];
                        sum2 = sum2 + Kernel2[j + 1][k + 1] * d_imgGray[x + YoffSet * j + k*3 + f];
                    }
                }
                //Segun dicta el algoritmo se aplica la siguiente operacion
                fsum = ceilf(sqrt((sum * sum) + (sum2 * sum2)));
                //el valor resultante se substituye en el pixel correspondiente de la imagen objetivo
                //printf("sum  %f, sum2: %f\n", sum, sum2);
                //d_imgSobel[x+f] = fsum;
            }
        }
    }
    else
    {
        printf("No Entro %i, inicio: %i y final %i\n", index, initIteration, endIteration);
    }

    __syncthreads();
}

//Se encarga de recibida dos imagenes de mismo tamaño, sobrescribir una como
//la version en escala de grises de la otra
void greyscaleProcessing(Mat imgOrig, Mat imgGray)
{
    //Se recorre toda la imagen original

    //#pragma omp parallel for collapse(2)

    for (int y = 0; y < imgOrig.rows; y++)
    {
        for (int x = 0; x < imgOrig.cols; x++)
        {
            //por cada posicion se realiza la siguiente operacion de pesos
            float gray = imgOrig.at<Vec3b>(y, x)[0] * 0.114 + imgOrig.at<Vec3b>(y, x)[1] * 0.587 + imgOrig.at<Vec3b>(y, x)[2] * 0.299;
            //El resultado se aplica a cada seccion RGB del pixel, para que se obtenga un gris acorde en ese punto
            imgGray.at<Vec3b>(y, x)[0] = gray;
            imgGray.at<Vec3b>(y, x)[1] = gray;
            imgGray.at<Vec3b>(y, x)[2] = gray;
        }
    }
}

//aca no es necesario pasarlas de a 3 pff jaja. esto seria para sobel
__global__ void gray(unsigned char *d_imgOrig, unsigned char *d_imgGray, int rows, int numberElements, int totalThreads)
{

    int x;

    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    int initIteration = (numberElements / totalThreads) * index;
    int endIteration = initIteration + (numberElements / totalThreads) - 1;

    if (endIteration < numberElements)
    {
        printf("Entro %i, inicio: %i y final %i\n", index, initIteration, endIteration);
        for (x = initIteration; x < endIteration; x = x + 3)
        {
            unsigned char r = d_imgOrig[x + 0];
            unsigned char g = d_imgOrig[x + 1];
            unsigned char b = d_imgOrig[x + 2];

            d_imgGray[x + 0] = r * 0.299f + g * 0.587f + b * 0.114f;
            d_imgGray[x + 1] = r * 0.299f + g * 0.587f + b * 0.114f;
            d_imgGray[x + 2] = r * 0.299f + g * 0.587f + b * 0.114f;
        }
    }
    else
    {
        printf("No Entro %i, inicio: %i y final %i\n", index, initIteration, endIteration);
    }

    __syncthreads();
}

int main(int argc, char *argv[])
{
    //-----------------------------------Variables------------------------------------//
    //errores de cuda
    cudaError_t err = cudaSuccess;
    int blocksPerGrid, threadsPerBlock;
    blocksPerGrid = 2;
    //blocksPerGrid = argv[3];
    threadsPerBlock = 6 / blocksPerGrid;
    //threadsPerBlock = argv[4];
    int totalThreads = blocksPerGrid * threadsPerBlock;
    //Definimos el conjunto de variables que utilizaremos para manejar las imagenes
    //Esto gracias al tipo de dato Mat que permite manejar la imagen como un objeto con atributos
    Mat imgOrig, imgSobel;
    unsigned char *h_imgOrig, *h_imgSobel, *h_imgGray;
    unsigned char *d_imgOrig, *d_imgSobel, *d_imgGray;
    int rows; //number of rows of pixels
    int cols; //number of columns of pixels
    //--------------------------------------------------------------------------------//

    //-----------------------------------Lectura imagen------------------------------------//
    //Se carga la imagen original como una imagen a color
    imgOrig = imread("128p.png", IMREAD_COLOR);
    //imgOrig = imread(argv[1], IMREAD_COLOR);

    //Se verifica que se cargo correctamente
    if (!imgOrig.data)
    {
        return -1;
    }

    //--------------------------------------------------------------------------------//

    //-----------------------------------Malloc------------------------------------//
    rows = imgOrig.rows;
    cols = imgOrig.cols;

    h_imgOrig = (unsigned char *)malloc(rows * cols * sizeof(unsigned char) * 3);
    unsigned char *rgb_image = imgOrig.data;

    //populate host's rgb data array
    int x = 0;
    for (x = 0; x < rows * cols * 3; x++)
    {
        h_imgOrig[x] = rgb_image[x];
    }

    size_t numElements = imgOrig.rows * imgOrig.cols;

    cv::Mat testData(rows, cols, CV_8UC3, (void *)h_imgOrig);
    //write Mat object to file
    cv::imwrite("TEST.jpg", testData);

    h_imgSobel = (unsigned char *)malloc(rows * cols * sizeof(unsigned char *)*3);
    h_imgGray = (unsigned char *)malloc(rows * cols * sizeof(unsigned char *)*3);
    //--------------------------------------------------------------------------------//

    //-----------------------------------CudaMalloc------------------------------------//

    err = cudaMalloc(&d_imgOrig, sizeof(unsigned char) * numElements * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector imgOrig (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_imgSobel, sizeof(unsigned char) * numElements * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector vector imgSobel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_imgSobel, 0, sizeof(unsigned char) * numElements * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set memory device vector imgSobel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_imgGray, sizeof(unsigned char) * numElements * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector imgGray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(d_imgGray, 0, sizeof(unsigned char) * numElements * 3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set memory device vector imgGray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //--------------------------------------------------------------------------------//

    //-----------------------------------Tiempo------------------------------------//
    //Establecemos las variables de tiempo para las mediciones respectivas
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    //--------------------------------------------------------------------------------//

    //-----------------------------------CudaMemcpy------------------------------------//
    err = cudaMemcpy(d_imgOrig, h_imgOrig, sizeof(unsigned char) * numElements * 3, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector imgOrig from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //--------------------------------------------------------------------------------//

    //-----------------------------------__Global__------------------------------------//
    //Se hace llamado al metodo encargado de pasar la imagen original a escala de grises
    //como paso fundamental antes de aplicar sobel

    //greyscaleProcessing(imgOrig, imgGray);
    printf("Tamano imagen %li, tamano *3: %li y hilos %i\n", numElements, numElements * 3, totalThreads);
    gray<<<blocksPerGrid, threadsPerBlock>>>(d_imgOrig, d_imgGray, rows, numElements * 3, totalThreads);
    //--------------------------------------------------------------------------------//

    //-----------------------------------CudaMemcpy - Results------------------------------------//
    err = cudaMemcpy(h_imgGray, d_imgGray, sizeof(unsigned char) * numElements * 3, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector imgGray from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //--------------------------------------------------------------------------------//

    printf("--------------------------------GrayscaleDONE------------------------------------\n");
    printf("--------------------------------GrayscaleDONE------------------------------------\n");
    printf("--------------------------------GrayscaleDONE------------------------------------\n");

    //-----------------------------------WriteGreyImg------------------------------------//
    //nombre de la imagen en escala de grises
    string string1("128p.png");
    string1 = string1.substr(0, string1.size() - 4);
    string1 += "grayscale.png";

    cv::Mat greyData(rows, cols, CV_8UC3, (void *)h_imgGray);
    //write Mat object to file
    cv::imwrite(string1, greyData);

    /*for (x = 0; x < cols * 3; x++)
	{
		h_imgSobel[x] = h_imgGray[x];
        h_imgSobel[x + cols*3]=255;
        h_imgSobel[x + cols*2*3]=200;
        h_imgSobel[x + cols*3*3]=150;
        h_imgSobel[x + cols*4*3]=100;
        h_imgSobel[x + cols*5*3]=50;
	}
    int offset = cols * 3 + 3;
    x = offset;
    h_imgSobel[x - offset + 0] = 255;
    h_imgSobel[x - offset + 1] = 255;
    h_imgSobel[x - offset + 2] = 255;
    h_imgSobel[x] = 255;
    h_imgSobel[x + 1] = 255;
    h_imgSobel[x + 2] = 255;

    cv::Mat sobelData(rows, cols, CV_8UC3, (void *)h_imgSobel);
    //write Mat object to file
    cv::imwrite("sobelTest.jpg", sobelData);*/
    //--------------------------------------------------------------------------------//

    //-----------------------------------Sobel - kernels------------------------------------//

    //Se definen los kernels para la operacion de sobel
    //uno que identifique los bordes verticales y uno para bordes horizontales
    float **Kernel;
    float **Kernel2;
    Kernel = (float **)malloc(3 * sizeof(float *));
    Kernel2 = (float **)malloc(3 * sizeof(float *));
    for (int i = 0; i < 3; i++)
    {
        Kernel[i] = (float *)malloc(3 * sizeof(float));
        Kernel2[i] = (float *)malloc(3 * sizeof(float));
    }
    Kernel[0][0] = -1;
    Kernel[0][1] = 0;
    Kernel[0][2] = 1;
    Kernel[1][0] = -2;
    Kernel[1][1] = 0;
    Kernel[1][2] = 2;
    Kernel[2][0] = -1;
    Kernel[2][1] = 0;
    Kernel[2][2] = 1;
    Kernel2[0][0] = -1;
    Kernel2[0][1] = -2;
    Kernel2[0][2] = -1;
    Kernel2[1][0] = 0;
    Kernel2[1][1] = 0;
    Kernel2[1][2] = 0;
    Kernel2[2][0] = 1;
    Kernel2[2][1] = 2;
    Kernel2[2][2] = 1;

    //--------------------------------------------------------------------------------//

    //-----------------------------------CudaMemcpy------------------------------------//
    /*err = cudaMemcpy(d_imgGray, h_imgGray, sizeof(unsigned char) * numElements * 3, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_imgGray from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
    //--------------------------------------------------------------------------------//

    //-----------------------------------__Global__------------------------------------//

    //Se llama a la funcion que realiza el procedimiento para hallar sobel
    sobel<<<blocksPerGrid, threadsPerBlock>>>(d_imgGray, d_imgSobel,cols, rows, numElements * 3, totalThreads,Kernel,Kernel2);
    //#pragma omp barrier
    for (int i = 0; i < 3; i++){
        free(Kernel[i]);
        free(Kernel2[i]);
    }
    free(Kernel);
    free(Kernel2);

    //--------------------------------------------------------------------------------//

    //-----------------------------------CudaMemcpy - Results------------------------------------//
    err = cudaMemcpy(h_imgSobel, d_imgSobel, sizeof(unsigned char) * numElements * 3, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector d_imgSobel from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //--------------------------------------------------------------------------------//

    //Se guarda la imagen correspondiente a sobel
    cv::Mat sobelData(rows, cols, CV_8UC3, (void *)h_imgSobel);
    //write Mat object to file
    cv::imwrite("sobelFinal.png", sobelData);
    //cv::imwrite(argv[2], sobelData);


    //-----------------------------------CudaFree------------------------------------//
    //al final
    err = cudaFree(d_imgOrig);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_imgOrig (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_imgGray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_imgGray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_imgSobel);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_imgSobel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //--------------------------------------------------------------------------------//

    //-----------------------------------Tiempo - Final------------------------------------//
    //Se finaliza el registro del tiempo
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    //escritura de los tiempos en el txt
    ofstream myfile;
    myfile.open("tiempos.txt", std::ios_base::app);
    myfile << "Imagen: "
           << "128p.png"
           << " - ";
    myfile << "Tiempo: " << tval_result.tv_sec << "." << tval_result.tv_usec << " s - ";
    myfile.close();

    printf("%ld.%ld \n", tval_result.tv_sec, tval_result.tv_usec);
    //---------------------------------------------------------------------------------//

    return 0;
}
