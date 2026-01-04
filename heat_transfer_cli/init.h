#pragma once

void initTopBottomTemperature(double *gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int nTopRows, unsigned int nBottomRows, double temperature)
{
    for (unsigned int i = 0; i < nTopRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = temperature;

    for (unsigned int i = nTopRows; i < nRows - nBottomRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = 0;

    for (unsigned int i = nRows - nBottomRows; i < nRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = temperature;
}
