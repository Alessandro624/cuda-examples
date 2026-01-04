#ifndef IO_H
#define IO_H

#include <iomanip>
#include <iostream>
#include <fstream>

void streamTemperature(std::ostream &file, unsigned int step, double *gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    for (unsigned int i = 0; i < nRows; i++)
    {
        for (unsigned int j = 0; j < nCols; j++)
            file << std::setw(fieldW) << gridTemperature[i * nCols + j] << " ";
        file << std::endl;
    }
    file << std::endl;
}

void printTemperature(unsigned int step, double *gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    streamTemperature(std::cout, step, gridTemperature, nRows, nCols, fieldW);
}

void saveTemparature(std::string file_base_name, std::string file_extension, unsigned int step, double *gridTemperature, unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    std::string filename = file_base_name + "_step_" + std::to_string(step) + file_extension;
    std::fstream file(filename, std::ios::out);
    if (file.is_open())
        streamTemperature(file, step, gridTemperature, nRows, nCols, fieldW);
    else
        file << "Unable to open file " << filename << "!" << std::endl;
    file.close();
}

#endif
