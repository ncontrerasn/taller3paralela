#include <iostream>
#include <cmath>
#include <fstream>
#include <sys/time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;
using std::ofstream;

//Se encarga de realiza el algoritmo de sobel a la imagen
void noBorderProcessing(Mat imgGray, Mat imgSobel, float **Kernel, float **Kernel2)
{
    float sum;
    float sum2;
    float fsum;
    //Se recorre la imagen completa exclugendo los pixeles borde para que la
    //operacion de convolucion siguiente puede hacerse sin padding
    for (int y = 1; y < imgGray.rows - 1; y++)
    {
        for (int x = 1; x < imgGray.cols - 1; x++)
        {
            //Se debe realizar la operacion por cada uno de los colores RGB que se encuentran en cada pixel
            for (int f = 0; f < 3; f++)
            {
                sum = 0.0;
                sum2 = 0.0;
                fsum = 0.0;
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
    float gray = 0.0;
    //Se recorre toda la imagen original
    for (int y = 0; y < imgOrig.rows; y++)
    {
        for (int x = 0; x < imgOrig.cols; x++)
        {
            //por cada posicion se realiza la siguiente operacion de pesos
            gray = imgOrig.at<Vec3b>(y, x)[0] * 0.114 + imgOrig.at<Vec3b>(y, x)[1] * 0.587 + imgOrig.at<Vec3b>(y, x)[2] * 0.299;
            //El resultado se aplica a cada seccion RGB del pixel, para que se obtenga un gris acorde en ese punto
            imgGray.at<Vec3b>(y, x)[0] = gray;
            imgGray.at<Vec3b>(y, x)[1] = gray;
            imgGray.at<Vec3b>(y, x)[2] = gray;
        }
    }
}

int main(int argc, char *argv[])
{
    //Definimos el conjunto de variables que utilizaremos para manejar las imagenes
    //Esto gracias al tipo de dato Mat que permite manejar la imagen como un objeto con atributos
    Mat imgOrig, imgSobel, imgGray;

    //Establecemos las variables de tiempo para las mediciones respectivas
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    
    int numThreads = argv[3];

    //Se carga la imagen original como una imagen a color
    imgOrig = imread(argv[1], IMREAD_COLOR);

    //Se verifica que se cargo correctamente
    if (!imgOrig.data)
    {
        return -1;
    }

    //Se hace una copia de la imagen original para luego pasarla a escala de grises
    imgGray = imgOrig.clone();

    //Se hace llamado al metodo encargado de pasar la imagen original a escala de grises
    //como paso fundamental antes de aplicar sobel
    greyscaleProcessing(imgOrig, imgGray);

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
    myfile << "Tiempo: " << tval_result.tv_sec << "." << tval_result.tv_usec << " s\n";
    myfile.close();

    //mostrar las imagenes de entrada y salida
    namedWindow("final");
    imshow("final", imgSobel);

    namedWindow("initial");
    imshow("initial", imgOrig);

    waitKey();

    return 0;
}
