#include "init.h"
#include "io.h"
#include "update.h"
#include "util.h"
#include <iostream>

int main()
{
    unsigned int step = 0;
    unsigned int nSteps{10000};
    unsigned int gridRows{1 << 8};
    unsigned int gridCols{1 << 12};
    unsigned int nHotTopRows{2};
    unsigned int nHotBottomRows{2};
    double initialHotTemperature{20};
    double *temperatureCurrent = new (std::nothrow) double[gridRows * gridCols];
    double *temperatureNext = new (std::nothrow) double[gridRows * gridCols];
    double elapsedTime{0.0};
    unsigned int fieldWidth{5};
    std::string outfilePrefix{"temperature"};
    std::string outfileExtension{".dat"};

    initTopBottomTemperature(temperatureCurrent, gridRows, gridCols, nHotTopRows, nHotBottomRows, initialHotTemperature);
    initTopBottomTemperature(temperatureNext, gridRows, gridCols, nHotTopRows, nHotBottomRows, initialHotTemperature);

    std::cout << "Saving initial configuration... " << std::endl;
    saveTemparature(outfilePrefix, outfileExtension, step, temperatureCurrent, gridRows, gridCols, fieldWidth);
    std::cout << "Done" << std::endl;

    std::cout << "Simulation in progress... " << std::endl;
    util::Timer clTimer;
    for (step = 1; step <= nSteps; step++)
    {
        updateRegion(temperatureNext, temperatureCurrent,
                     gridRows, gridCols,
                     nHotTopRows, (gridRows - 1) - nHotBottomRows,
                     1, (gridCols - 1) - 1);

        swapBufferPtrs(temperatureNext, temperatureCurrent);
    }
    elapsedTime = static_cast<double>(clTimer.getTimeMilliseconds());
    std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms (corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

    std::cout << "Saving final configuration... " << std::endl;
    saveTemparature(outfilePrefix, outfileExtension, --step, temperatureCurrent, gridRows, gridCols, fieldWidth);
    std::cout << "Done" << std::endl;

    /*
     * To visualize the simulation outcome, run gnuplot and use the following command:
     *
     *   plot 'temperature_step_N.dat' matrix with image
     *
     * where N is the step of the final configuration. Use quit to exit gnuplot.
     *
     */

    delete[] temperatureCurrent, temperatureNext;
    return 0;
}
