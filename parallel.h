#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>


__global__ void GetNearestCentroid(int* d_ret, float* d_xs, float* d_ys, float* d_zs, float* d_centroidX, float* d_centroidY, float* d_centroidZ, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // "Max float"
    float minDist = 1000.0f;
    int minDistInd = 0;

    if (i < n)
    {
        float x = d_xs[i];
        float y = d_ys[i];
        float z = d_zs[i];

        for (int j = 0; j<k; j++)
        {
            float distX = d_centroidX[j] - x;
            float distY = d_centroidY[j] - y;
            float distZ = d_centroidZ[j] - z;


            float curDist = sqrt(distX * distX + distY * distY + distZ * distZ);

            if (x == 2.5f)
            {

            }

            if (curDist<minDist)
            {
                minDist = curDist;
                minDistInd = j;
            }
        }

        //printf("For point x: %f, y: %f, z: %f closest centroid is: %d with dist %f \n", x, y, z, minDistInd, minDist);

        d_ret[i] = minDistInd;
    }
}


__global__ void CountOccurences(int* d_ret, float* d_xs, float* d_ys, float* d_zs, float* d_sumForCentroidX, float* d_sumForCentroidY, float* d_sumForCentroidZ, float* d_countForCentroid, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        int clas = d_ret[i];

        atomicAdd(&d_sumForCentroidX[clas], d_xs[i]);
        atomicAdd(&d_sumForCentroidY[clas], d_ys[i]);
        atomicAdd(&d_sumForCentroidZ[clas], d_zs[i]);

        atomicAdd(&d_countForCentroid[clas], 1.0f);
    }

}


// k - number of classes
// n - number of examples
int* kMeansParallel(int k, int n, float* xs, float* ys, float* zs, float* startCentroidX, float* startCentroidY, float* startCentroidZ, int printCentroids)
{
    int* d_ret;
    cudaMalloc((void**)&d_ret, n * sizeof(int));

    float* d_xs;
    cudaMalloc((void**)&d_xs, n * sizeof(float));
    cudaMemcpy(d_xs, xs, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_ys;
    cudaMalloc((void**)&d_ys, n * sizeof(float));
    cudaMemcpy(d_ys, ys, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_zs;
    cudaMalloc((void**)&d_zs, n * sizeof(float));
    cudaMemcpy(d_zs, zs, n * sizeof(float), cudaMemcpyHostToDevice);

    float* h_centroidX = new float[k];
    float* d_centroidX;
    cudaMalloc((void**)&d_centroidX, k * sizeof(float));
    cudaMemcpy(d_centroidX, startCentroidX, k * sizeof(float), cudaMemcpyHostToDevice);

    float* h_centroidY = new float[k];
    float* d_centroidY;
    cudaMalloc((void**)&d_centroidY, k * sizeof(float));
    cudaMemcpy(d_centroidY, startCentroidY, k * sizeof(float), cudaMemcpyHostToDevice);

    float* h_centroidZ = new float[k];
    float* d_centroidZ;
    cudaMalloc((void**)&d_centroidZ, k * sizeof(float));
    cudaMemcpy(d_centroidZ, startCentroidZ, k * sizeof(float), cudaMemcpyHostToDevice);


    // To count next centroids
    float* d_sumForCentroidsX;
    cudaMalloc((void**)&d_sumForCentroidsX, k * sizeof(float));

    float* d_sumForCentroidsY;
    cudaMalloc((void**)&d_sumForCentroidsY, k * sizeof(float));

    float* d_sumForCentroidsZ;
    cudaMalloc((void**)&d_sumForCentroidsZ, k * sizeof(float));

    float* d_countForCentroids;
    cudaMalloc((void**)&d_countForCentroids, k * sizeof(float));



    int blockSize = 512;
    int numberOfBlocks = n / blockSize;

    if (numberOfBlocks * blockSize < n)
    {
        numberOfBlocks++;
    }

    float* h_sumForCentroidsX = new float[k];
    float* h_sumForCentroidsY = new float[k];
    float* h_sumForCentroidsZ = new float[k];
    float* h_countsForCentroids = new float[k];


    // TODO: Continue till there are changes of classes (say, 0.01 of all samples is changing class)
    for (int i=0; i<2; i++)
    {
        GetNearestCentroid<<<numberOfBlocks, blockSize>>>(d_ret, d_xs, d_ys, d_zs, d_centroidX, d_centroidY, d_centroidZ, n, k);

        cudaMemset(d_sumForCentroidsX, 0, k * sizeof(float));
        cudaMemset(d_sumForCentroidsY, 0, k * sizeof(float));
        cudaMemset(d_sumForCentroidsZ, 0, k * sizeof(float));
        cudaMemset(d_countForCentroids, 0, k * sizeof(float));

        CountOccurences<<<numberOfBlocks, blockSize>>>(d_ret, d_xs, d_ys, d_zs, d_sumForCentroidsX, d_sumForCentroidsY, d_sumForCentroidsZ, d_countForCentroids, n);

        cudaMemcpy(h_sumForCentroidsX, d_sumForCentroidsX, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sumForCentroidsY, d_sumForCentroidsY, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sumForCentroidsZ, d_sumForCentroidsZ, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_countsForCentroids, d_countForCentroids, k * sizeof(float), cudaMemcpyDeviceToHost);

        if (printCentroids) {
            printf("\n");
        }

        for (int j=0; j<k; j++) {
            float sum = h_countsForCentroids[j];

            h_centroidX[j] = h_sumForCentroidsX[j] / sum;
            h_centroidY[j] = h_sumForCentroidsY[j] / sum;
            h_centroidZ[j] = h_sumForCentroidsZ[j] / sum;

            if (printCentroids) {
                printf("Centroid: %f %f %f \n", h_centroidX[j], h_centroidY[j], h_centroidZ[j]);
            }
        }

        cudaMemcpy(d_centroidX, h_centroidX, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroidY, h_centroidY, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroidZ, h_centroidZ, k * sizeof(float), cudaMemcpyHostToDevice);
    }


    int* h_ret = new int[n];
    cudaMemcpy(h_ret, d_ret, n * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Parallel result\n");
    for (int i=0; i<n; i++)
    {
        printf("%d ", h_ret[i]);
    }

    cudaFree(d_ret);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_centroidX);
    cudaFree(d_centroidY);
    cudaFree(d_centroidZ);
    cudaFree(d_sumForCentroidsX);
    cudaFree(d_sumForCentroidsY);
    cudaFree(d_sumForCentroidsZ);
    cudaFree(d_countForCentroids);


    return h_ret;
}