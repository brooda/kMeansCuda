#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "classic.h"
#include "parallel.h"

float randFloat(float a) {
    return ((float) rand() / (float) (RAND_MAX)) * a;
}

void usage()
{
    printf("Required 2 input parameter: file length and k (number of clusters)");
}

int main(int argc, char **argv) {
    if (argc != 3)
    {
        usage();
    }

    int cardinality = atoi(argv[1]);
    int k = atoi(argv[2]);

    char* filePath = new char[30];
    sprintf(filePath, "input/%d/%d.csv", k, cardinality);

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    FILE* fp = fopen(filePath, "r");

    float* xs = new float[cardinality];
    float* ys = new float[cardinality];
    float* zs = new float[cardinality];

    int i=0;
    while ((read = getline(&line, &len, fp)) != -1) {
        float x;
        float y;
        float z;

        sscanf(line, "%f,%f,%f", &x,&y,&z );

        xs[i] = x;
        ys[i] = y;
        zs[i++] = z;
    }

    fclose(fp);

    float* startCentroidX = new float[k];
    float* startCentroidY = new float[k];
    float* startCentroidZ = new float[k];

    srand(2);
    for (int i=0; i<k; i++)
    {
        startCentroidX[i] = randFloat(4);
        startCentroidY[i] = randFloat(4);
        startCentroidZ[i] = randFloat(4);
    }
    printf("\n");

    clock_t start, end;
    double cpu_time_used;


    int printCentroids = 0;

    start = clock();
    int* classes = kMeans(k, cardinality, xs, ys, zs, startCentroidX, startCentroidY, startCentroidZ, printCentroids);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Sequential took %f seconds to execute \n", cpu_time_used);

    start = clock();
    int* classesParallel = kMeansParallel(k, cardinality, xs, ys, zs, startCentroidX, startCentroidY, startCentroidZ, printCentroids);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Parallel took %f seconds to execute \n", cpu_time_used);

    int mismatched = 0;
    for(int i=0; i<cardinality; i++)
    {
        if (classes[i] != classesParallel[i])
        {
            mismatched++;
        }
    }
    printf("Number of mismatched: %d \n", mismatched);

    char* outputFile = new char[30];
    sprintf(outputFile, "results/%d.txt", cardinality);
    FILE *f = fopen(outputFile, "ab+");

    for (int i=0; i<cardinality - 1; i++)
    {
        fprintf(f,"%d,", classes[i]);
    }

    fprintf(f,"%d", classes[cardinality - 1]);
    fclose(f);

    return 0;
}
