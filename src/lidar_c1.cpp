#include "lidar_c1.hpp"
#include <cmath>
#include <stdexcept>

using namespace rp::standalone::rplidar;

static inline float deg2rad(float deg) {
    return deg * 3.14159265f / 180.0f;
}

// Convert [0,360) -> [-180,180)
static inline float wrap_deg_pm180(float deg0_360) {
    return (deg0_360 >= 180.0f) ? (deg0_360 - 360.0f) : deg0_360;
}

LidarC1::LidarC1(const std::string& port, int baud) {
    // creates sdk driver object drv_
    drv_ = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
    // this is just incase it cant create the driver properly 
    if (!drv_) throw std::runtime_error("Failed to create RPlidarDriver");
    // try to connect to the lidar on the specified port and at the baud rate
    // port.c_str converts cpp string into char because thats what sdk wants 
    // sl_u32 casts baud to int as thats what sdk wants 
    // IS_FAIL checks if sdk fucntions retruned any error 
    if (IS_FAIL(drv_->connect(port.c_str(), (sl_u32)baud))) {
        RPlidarDriver::DisposeDriver(drv_); // if it does error free the driver cleanly 
        drv_ = nullptr; // null it 
        throw std::runtime_error("Failed to connect to LiDAR on " + port);
    }

    drv_->startMotor(); // start spinning the motor
    drv_->startScan(0, 1); // start scanning 
}
// Destructor to stop and clean 
LidarC1::~LidarC1() {
    if (drv_) {
        drv_->stop();
        drv_->stopMotor();
        drv_->disconnect();
        RPlidarDriver::DisposeDriver(drv_);
        drv_ = nullptr;
    }
}
//retruns a vector of angle_rad, distance_m pairs
// if driver not available just let it retrun and empty vector 
std::vector<LidarPoint> LidarC1::getScan() {
    std::vector<LidarPoint> scan;
    if (!drv_) return scan;

    sl_lidar_response_measurement_node_hq_t nodes[8192]; // big fixed array to hold raw lidar points 
    size_t count = sizeof(nodes) / sizeof(nodes[0]); // count starts as max cap 8192

    // asks skd to fill nodes with latest scan 
    // on return, count becomes how many points were actually filled
    // if it fails retruns and empty vector 
    if (IS_FAIL(drv_->grabScanDataHq(nodes, count))) return scan;
    drv_->ascendScanData(nodes, count); // sorts the point by angle increasing to make later searching easier 


    // convert raw sdk units into useful units 
    // pre allocate vector mem
    scan.reserve(count);
    for (size_t i = 0; i < count; i++) { // loop over eahc measurement point 
        // angle_z_q14 is from sdk, fixed point angle format 
        // this line converts it into degrees 0-360
        float angle_deg0_360 = (nodes[i].angle_z_q14 * 90.f) / 16384.f; 
        float angle_deg = wrap_deg_pm180(angle_deg0_360); //wraps to -180 > +180
        float angle_rad = deg2rad(angle_deg); // converted to radians 
        
        // distance comes as mm so divide by 1000 to get metres 
        float dist_mm = nodes[i].dist_mm_q2 / 4.0f;
        float dist_m  = dist_mm / 1000.0f;

        if (dist_m <= 0.05f) continue; // ignore any tiny distances 
        scan.emplace_back(angle_rad, dist_m); // push the angle_rad, distance m pair into the output vector 
    }
    return scan; 
}
