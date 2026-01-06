#pragma once // only include this header once per comp
#include <string>
#include <utility>
#include <vector>

#include "rplidar.h"  // Slamtec SDK

using LidarPoint = std::pair<float, float>; // (angle_rad, distm)

class LidarC1 {
public:
    LidarC1(const std::string& port, int baud); // constructor declaration
    ~LidarC1(); // destrcutor declaration
    std::vector<LidarPoint> getScan();

private:
    rp::standalone::rplidar::RPlidarDriver* drv_ = nullptr;
};

