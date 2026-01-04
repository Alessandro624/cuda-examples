#pragma once

void updateTemperatureKernel(double *TNext, double *T, unsigned int i, unsigned int j, unsigned int nCols)
{
  TNext[i * nCols + j] = (1.0 / 20.0) *
                         (4 * (T[i * nCols + (j + 1)] +
                               T[i * nCols + (j - 1)] +
                               T[(i + 1) * nCols + j] +
                               T[(i - 1) * nCols + j]) +
                          T[(i + 1) * nCols + (j + 1)] +
                          T[(i + 1) * nCols + (j - 1)] +
                          T[(i - 1) * nCols + (j + 1)] +
                          T[(i - 1) * nCols + (j - 1)]);
}
