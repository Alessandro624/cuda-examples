#pragma once

#include "kernel.h"

void updateRegion(double *next, double *curr,
                  unsigned int nRows,
                  unsigned int nCols,
                  unsigned int i_start,
                  unsigned int i_end,
                  unsigned int j_start,
                  unsigned int j_end)
{
    for (unsigned int i = i_start; i <= i_end; i++)
        for (unsigned int j = j_start; j <= j_end; j++)
            updateTemperatureKernel(next, curr, i, j, nCols);
}

void swapBufferPtrs(double *&next, double *&curr)
{
    double *tempPtr = next;
    next = curr;
    curr = tempPtr;
}
