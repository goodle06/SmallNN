#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <iostream>
#include <string>

#include <vector>
#include <functional>
#include <numeric>
#include <execution>
#include <cmath>
#include <optional>
#include <random>
#include <ctime>
#include <limits>
#include <regex>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <future>
#include <mutex>

#include <mkl.h>

#include <Proj/WrappersAndServices/WindowsWrappers.h>

/*Third party */
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>

#define LOG std::cout
#define TIMER_START std::chrono::steady_clock::now()
#define TIMER_DURATION(a) std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count()/60.0
#define TIMER_END(a) LOG << std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count()/60.0 << "\n"

#define RUN_PARALLEL 0
#define DISPLAY_PROGRESS 1

#endif // COMMON_H