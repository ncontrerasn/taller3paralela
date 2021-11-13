#include<iostream>
#include <cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;


void noBorderProcessing(Mat src, Mat dst, float Kernel[][3], float Kernel2[][3])
{

    float sum;
    float sum2;
    float fsum;
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int f = 0; f<3; f++) {
            sum = 0.0;
            sum2 = 0.0;
            fsum = 0.0;
            for (int k = -1; k <= 1; k++) {
                for (int j = -1; j <= 1; j++) {
                        sum = sum + Kernel[j + 1][k + 1] * src.at<Vec3b>(y - j, x - k)[f];
                        sum2 = sum2 + Kernel2[j + 1][k + 1] * src.at<Vec3b>(y - j, x - k)[f];
                    }
                    
                }
            fsum=ceil(sqrt((sum * sum) + (sum2 * sum2)));
            dst.at<Vec3b>(y, x)[f] = fsum;
            }
        }
    }
}

void greyscaleProcessing(Mat src, Mat gry)
{
    float gray = 0.0;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            gray = src.at<Vec3b>(y,x)[0] * 0.114 + src.at<Vec3b>(y,x)[1]*0.587 + src.at<Vec3b>(y,x)[2]*0.299;
            gry.at<Vec3b>(y,x)[0] = gray;
            gry.at<Vec3b>(y,x)[1] = gray;
            gry.at<Vec3b>(y,x)[2] = gray;
        }
    }
}

int main(int argc, char *argv[])
{
    //Definicion de las variables de imagenes
    Mat src, dst, gry;

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    //Carga de la imagen principal
    src = imread(argv[1], IMREAD_COLOR);

    if (!src.data)
    {
        return -1;
    }

    gry = imread(argv[1]);

    greyscaleProcessing(src,gry);

    imwrite("grayscale.png", gry);

    float Kernel[3][3] = {
                          {-1, 0, 1},
                          {-2, 0, 2},
                          {-1, 0, 1}
    };
    float Kernel2[3][3] = {
                          {-1, -2, -1},
                          {0, 0, 0},
                          {1, 2, 1}
    };

    dst = gry.clone();
    for (int y = 0; y < gry.rows; y++)
        for (int x = 0; x < gry.cols; x++)
            dst.at<uchar>(y, x) = 0.0;

    noBorderProcessing(gry, dst, Kernel, Kernel2);

    imwrite(argv[2], dst);

    gettimeofday(&tval_after, NULL);
   
    timersub(&tval_after, &tval_before, &tval_result);
   
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec,(long int)tval_result.tv_usec);
    namedWindow("final");
    imshow("final", dst);

    namedWindow("initial");
    imshow("initial", src);

    waitKey();

    return 0;
}

