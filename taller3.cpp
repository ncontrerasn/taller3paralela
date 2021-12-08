#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/time.h>
#include <omp.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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
__global__ void gray(Mat *imgOrig, Mat *imgGray){
    __shared__ (tipo de dato Vec3b, creo) lineasOrig[3 * imgOrig.cols * sizeof(un lemento de Mat)];
    __shared__ lineasGray[3 * imgOrig.cols * sizeof(un lemento de Mat)];

    int z = 0;

    //copiar las 3 filas de la original y gris en memoria compartida
    //para saber cuales 3 filas copiar, hay que saber el id del bloque
    for (int y = 0; y < 3; y++)
    {
        for (int x = 0; x < imgOrig.cols; x++)
        {
            lineasOrig[z] = imgOrig.at<Vec3b>(y, x)[x + blockIdx.x * imgOrig.cols];//me perdi a
            lineasGary[z] = imgGray[x + blockIdx.x * imgGray.cols];          
        }
        z++;
    }

    for (int x = 0; x < lineasOrig.size; x++)
    {
        //por cada posicion se realiza la siguiente operacion de pesos
        //aplicar el efecto pero ya no es y,x sino solo x
        float gray = lineasOrig.at<Vec3b>(x)[0] * 0.114 + lineasOrig.at<Vec3b>(x)[1] * 0.587 + lineasOrig.at<Vec3b>(x)[2] * 0.299;
        //El resultado se aplica a cada seccion RGB del pixel, para que se obtenga un gris acorde en ese punto
        lineasGray.at<Vec3b>(x)[0] = gray;
        lineasGray.at<Vec3b>(x)[1] = gray;
        lineasGray.at<Vec3b>(x)[2] = gray;
    }
    //pasar los de las 3 lineas a la memoria general? o saltarse esto y hacer el paso anterior directo
    //a la memoria general

    for (int x = 0; x < lineasOrig.size; x++)
    {
        imgGray[x + blockIdx.x * imgGray.cols];
        lineasGray.at<Vec3b>(x)[0] = gray;
        lineasGray.at<Vec3b>(x)[1] = gray;
        lineasGray.at<Vec3b>(x)[2] = gray;
    }
    
}

int main(int argc, char *argv[])
{
    //son constantes para cuaquier tamaño de imagen?
    int bloques = 1, hilos = 1;
    //Definimos el conjunto de variables que utilizaremos para manejar las imagenes
    //Esto gracias al tipo de dato Mat que permite manejar la imagen como un objeto con atributos
    Mat imgOrig, imgSobel, imgGray;
    Mat *d_imgOrig, *d_imgSobel, *d_imgGray;

    size = sizeof(Mat);

    cudaMalloc((void **) &d_imgOrig, size);
    cudaMalloc((void **) &d_imgSobel, size);
    cudaMalloc((void **) &d_imgGray, size);

    //Establecemos las variables de tiempo para las mediciones respectivas
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    
    /*
    int numThreads = atoi(argv[3]);
    printf("%d\n",numThreads);
    omp_set_num_threads(numThreads);
    */

    //Se carga la imagen original como una imagen a color
    imgOrig = imread(argv[1], IMREAD_COLOR);

    //Se verifica que se cargo correctamente
    if (!imgOrig.data)
    {
        return -1;
    }

    //Se hace una copia de la imagen original para luego pasarla a escala de grises
    imgGray = imgOrig.clone();

    cudaMemcpy(d_imgOrig, &imgOrig, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgGray, &imgGray, size, cudaMemcpyHostToDevice);

    //Se hace llamado al metodo encargado de pasar la imagen original a escala de grises
    //como paso fundamental antes de aplicar sobel
    
    //greyscaleProcessing(imgOrig, imgGray);

    gray<<bloques, hilos>>(d_imgOrig, d_imgGray);

    cudaMemcpy(&imgGray, d_imgGray, size, cudaMemcpyDeviceToHost);

    //al final 
    cudaFree(d_imgOrig);
    cudaFree(d_imgGray);
    cudaFree(d_imgSobel);
    
    //#pragma omp barrier

    //nombre de la imagen en escala de grises
    string string1(argv[1]);
    string1 = string1.substr(0, string1.size() - 4);
    string1 += "grayscale.png";

    //Se guarda la imagen en escala de grises
    imwrite(string1, imgGray);

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

    //Se vuelve a hacer una copia, en este caso, para tener una imagen donde registrar el sobel
    //utilizar la misma variable ocacionaria errores en el procedimiento de sobel
    imgSobel = imgGray.clone();

    //Se llama a la funcion que realiza el procedimiento para hallar sobel
    noBorderProcessing(imgGray, imgSobel, Kernel, Kernel2);

    //#pragma omp barrier

    for (int i = 0; i < 3; i++){
        free(Kernel[i]);
        free(Kernel2[i]);
    }
    free(Kernel);
    free(Kernel2);

    //Se guarda la imagen correspondiente a sobel
    imwrite(argv[2], imgSobel);

    //Se finaliza el registro del tiempo
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    //escritura de los tiempos en el txt
    ofstream myfile;
    myfile.open("tiempos.txt", std::ios_base::app);
    myfile << "Imagen: " << argv[1] << " - ";
    myfile << "Tiempo: " << tval_result.tv_sec << "." << tval_result.tv_usec << " s - ";
     myfile << "Hilos: " << numThreads << "\n";
    myfile.close();

    printf("%ld.%ld \n",tval_result.tv_sec,tval_result.tv_usec);

    /*//mostrar las imagenes de entrada y salida
    namedWindow("final");
    imshow("final", imgSobel);

    namedWindow("initial");
    imshow("initial", imgOrig);

    waitKey();*/

    return 0;
}
