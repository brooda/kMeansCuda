#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>


int getIndexOfClosestCentroid(float x, float y, float z, float* centroidX, float* centroidY, float* centroidZ, int k, int last, int* changes)
{
    // Max float
    float minDist = 1000.0;
    int minDistInd = 0;

    for (int i=0; i<k; i++)
    {
        float distX = centroidX[i] - x;
        float distY = centroidY[i] - y;
        float distZ = centroidZ[i] - z;

        float curDist = sqrt( distX*distX + distY*distY + distZ*distZ );

        if (curDist < minDist)
        {
            minDist = curDist;
            minDistInd = i;
        }
    }


    if (last != minDistInd)
    {
        (*changes)++;
    }

    return minDistInd;
}


// k - number of classes
// n - number of examples
int* kMeans(int k, int n, float* xs, float* ys, float* zs, float* startCentroidX, float* startCentroidY, float* startCentroidZ, int printCentroids)
{
    int* ret = new int[n];

    float* centroidX = new float[k];
    float* centroidY = new float[k];
    float* centroidZ = new float[k];

    for(int i=0; i<k; i++)
    {
        centroidX[i] = startCentroidX[i];
        centroidY[i] = startCentroidY[i];
        centroidZ[i] = startCentroidZ[i];
    }

    int changes = n/100 + 1;

    while(changes > n/100)
    {
        changes = 0;

        for (int j=0; j<n; j++)
        {
            ret[j] = getIndexOfClosestCentroid(xs[j], ys[j], zs[j], centroidX, centroidY, centroidZ, k, ret[j], &changes);
        }

        float* sumForCentroidsX = new float[k];
        float* sumForCentroidsY = new float[k];
        float* sumForCentroidsZ = new float[k];
        float* countsForCentroids = new float[k];

        for (int j = 0; j<k; j++)
        {
            sumForCentroidsX[j] = 0;
            sumForCentroidsY[j] = 0;
            sumForCentroidsZ[j] = 0;
            countsForCentroids[j] = 0;
        }

        for (int j=0; j<n; j++)
        {
            int clas = ret[j];
            sumForCentroidsX[clas] += xs[j];
            sumForCentroidsY[clas] += ys[j];
            sumForCentroidsZ[clas] += zs[j];

            countsForCentroids[clas]+=1.0f;
        }

        if (printCentroids)
        {
            printf("\n");
        }

        for (int j=0; j<k; j++) {
            float sum = countsForCentroids[j];

            centroidX[j] = sumForCentroidsX[j] / sum;
            centroidY[j] = sumForCentroidsY[j] / sum;
            centroidZ[j] = sumForCentroidsZ[j] / sum;

            if (printCentroids) {
                printf("Centroid: %f %f %f \n", centroidX[j], centroidY[j], centroidZ[j]);
            }
        }
    }

    return ret;
}