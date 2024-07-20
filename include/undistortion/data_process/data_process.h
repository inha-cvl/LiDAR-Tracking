#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <fstream>
#include "gyr_int.h"
#include "sophus/se3.hpp"
#include <cmath>
#include <time.h>

#include "params.h"

typedef pcl::PointCloud<PointType> PointT;

inline double rad2deg(double radians) { return radians * 180.0 / M_PI; }
inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

struct MeasureGroup {
  sensor_msgs::PointCloud2ConstPtr lidar;
  std::vector<sensor_msgs::Imu::ConstPtr> imu;
};

class ImuProcess {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Process(const MeasureGroup &meas, PointT::Ptr &clout_out, double &time_taken);
  void Reset();

  void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu);

  void UndistortPcl(const pcl::PointCloud<PointXYZIT>::Ptr &pcl_in_out, double dt_be, const Sophus::SE3d &Tbe);
  void UndistortPcl(const pcl::PointCloud<ouster_ros::Point>::Ptr &pcl_in_out, double dt_be, const Sophus::SE3d &Tbe);
  

 private:
  /// Whether is the first frame, init for first frame
  bool b_first_frame_ = true;

  //// Input pointcloud
  PointT::Ptr cur_pcl_in_;
  //// Undistorted pointcloud
  PointT::Ptr cur_pcl_un_;

  double dt_l_c_;

  /// Transform form lidar to imu
  Sophus::SE3d T_i_l;
  //// For timestamp usage
  sensor_msgs::PointCloud2ConstPtr last_lidar_;
  sensor_msgs::ImuConstPtr last_imu_;
  Sophus::SO3d last_rotation_;

  /// For gyroscope integration
  GyrInt gyr_int_;
};

#endif  // LOAM_HORIZON_DATA_PROCESS_H
