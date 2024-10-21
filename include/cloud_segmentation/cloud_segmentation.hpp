#include "utils.hpp"
#include "patchworkpp/patchworkpp.hpp"

using ClusterPointT = pcl::PointXYZI;

template <typename PointT>
class CloudSegmentation {
public:
    CloudSegmentation() {};

    CloudSegmentation(ros::NodeHandle& nh) : nh_(nh) {
        nh_.getParam("Public/map", map);
        nh_.getParam("Public/lidar_frame", lidar_frame);
        nh_.getParam("Public/target_frame", target_frame);
        nh_.getParam("Public/world_frame", world_frame);
        nh_.getParam("Cloud_Segmentation/lidar_settings/V_SCAN", V_SCAN);
        nh_.getParam("Cloud_Segmentation/lidar_settings/H_SCAN", H_SCAN);
        nh_.getParam("Cloud_Segmentation/lidar_settings/ang_res_x", ang_res_x);
        nh_.getParam("Cloud_Segmentation/lidar_settings/ang_res_y", ang_res_y);
        nh_.getParam("Cloud_Segmentation/lidar_settings/ang_bottom", ang_bottom);
        nh_.getParam("Cloud_Segmentation/downsampling/leaf_size/x", leaf_size_x);
        nh_.getParam("Cloud_Segmentation/downsampling/leaf_size/y", leaf_size_y);
        nh_.getParam("Cloud_Segmentation/downsampling/leaf_size/z", leaf_size_z);
        nh_.getParam("Cloud_Segmentation/clustering/filter/min_size/x", filter_min_size_x);
        nh_.getParam("Cloud_Segmentation/clustering/filter/min_size/y", filter_min_size_y);
        nh_.getParam("Cloud_Segmentation/clustering/filter/min_size/z", filter_min_size_z);
        nh_.getParam("Cloud_Segmentation/clustering/filter/max_size/x", filter_max_size_x);
        nh_.getParam("Cloud_Segmentation/clustering/filter/max_size/y", filter_max_size_y);
        nh_.getParam("Cloud_Segmentation/clustering/filter/max_size/z", filter_max_size_z);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/min_size", adaptive_min_size);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/max_size", adaptive_max_size);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/min_tolerance", min_tolerance);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/max_tolerance", max_tolerance);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/max_region_distance", max_region_distance);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/number_region", number_region);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/min_leaf_size", min_leaf_size);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/max_leaf_size", max_leaf_size);
        nh_.getParam("Cloud_Segmentation/clustering/adaptive/thresh_iou", thresh_iou);
        nh_.getParam("Cloud_Segmentation/clustering/L_shape_fitting/projection_range", projection_range);
        nh_.getParam("Cloud_Segmentation/crop/max/x", roi_max_x);
        nh_.getParam("Cloud_Segmentation/crop/max/y", roi_max_y);
        nh_.getParam("Cloud_Segmentation/crop/max/z", roi_max_z);
        nh_.getParam("Cloud_Segmentation/crop/min/x", roi_min_x);
        nh_.getParam("Cloud_Segmentation/crop/min/y", roi_min_y);
        nh_.getParam("Cloud_Segmentation/crop/min/z", roi_min_z);
        nh_.getParam("Cloud_Segmentation/crop/crop_ring/enabled", crop_ring_enabled);
        nh_.getParam("Cloud_Segmentation/crop/crop_ring/ring", crop_ring);
        nh_.getParam("Cloud_Segmentation/crop/crop_intensity/enabled", crop_intensity_enabled);
        nh_.getParam("Cloud_Segmentation/crop/crop_intensity/intensity", crop_intensity);
        nh_.getParam("Cloud_Segmentation/crop/crop_hd_map/radius", crop_hd_map_radius);

        global_path = map_reader(map.c_str());

        // lidar
        dt_l_c_ = 0.05;
        cur_stamp = ros::Time(0);
        pre_stamp = ros::Time(0);

        // imu
        imu_cache.setCacheSize(1000);
        last_timestamp_imu = -1;
        Eigen::Quaterniond q(1, 0, 0, 0);
        Eigen::Vector3d t(0, 0, 0);
        T_i_l = Sophus::SE3d(q, t);

        // clustering
        delta_tolerance = (max_tolerance - min_tolerance) / number_region;

        // ground removal
        PatchworkppGroundSeg.reset(new PatchWorkpp<PointT>(&nh));

        // clear log
        clearLogFile(projection_time_log_path);
        clearLogFile(convert_time_log_path);
        clearLogFile(crop_time_log_path);
        clearLogFile(crophdmap_time_log_path);
        clearLogFile(removalground_time_log_path);
        clearLogFile(undistortion_time_log_path);
        clearLogFile(downsampling_time_log_path);
        clearLogFile(clustering_time_log_path);
        clearLogFile(lshape_time_log_path);
        clearLogFile(average_time_log_path);
    }

    void msgToPointCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg, pcl::PointCloud<PointT>& cloud);
    void imuUpdate(const sensor_msgs::Imu::ConstPtr &imu_msg);
    void projectPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    void convertPointCloudToImage(const pcl::PointCloud<PointT>& cloudIn, cv::Mat &imageOut, double &time_taken);
    void cropPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    void cropHDMapPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, 
                            tf2_ros::Buffer &tf_buffer, double &time_taken);
    void removalGroundPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    void undistortPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    
    // CUDA-PointPillars
    void pcl2FloatArray(const pcl::PointCloud<PointT>& cloudIn, std::vector<float>& arrayOut, double &time_taken);

    void downsamplingPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<ClusterPointT>& cloudOut, double &time_taken);
    void adaptiveClustering(const pcl::PointCloud<ClusterPointT>& cloudIn, std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double &time_taken);
    void voxelClustering(const pcl::PointCloud<PointT>& cloudIn, std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double& time_taken);
    void fittingLShape(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters,
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    // TODO
    void adaptiveVoxelClustering(const pcl::PointCloud<PointT>& cloudIn, 
                                                     std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, 
                                                     double& time_taken);
    void averageTime();

private:
    ros::NodeHandle nh_;
    std::string map;
    std::string lidar_frame;
    std::string target_frame;
    std::string world_frame;
    std::vector<std::pair<float, float>> global_path;
    int V_SCAN; // Vertical scan lines
    int H_SCAN; // Horizontal scan points per line
    float ang_res_x; // Angular resolution in x direction (degrees)
    float ang_res_y; // Angular resolution in y direction (degrees)
    int ang_bottom; // Bottom angle (degrees)

    // Downsampling parameters
    float leaf_size_x; // Leaf size for downsampling in x dimension
    float leaf_size_y; // Leaf size for downsampling in y dimension
    float leaf_size_z; // Leaf size for downsampling in z dimension

    // Clustering filter settings
    float filter_min_size_x; // Minimum size in x dimension for clustering
    float filter_min_size_y; // Minimum size in y dimension for clustering
    float filter_min_size_z; // Minimum size in z dimension for clustering
    float filter_max_size_x; // Maximum size in x dimension for clustering
    float filter_max_size_y; // Maximum size in y dimension for clustering
    float filter_max_size_z; // Maximum size in z dimension for clustering

    // Adaptive clustering settings
    int adaptive_min_size; // Minimum cluster size in points
    int adaptive_max_size; // Maximum cluster size in points
    float min_tolerance; // Start tolerance for clustering
    float max_tolerance; // Delta tolerance increment in clustering
    int max_region_distance; // Maximum regions for clustering
    int number_region; // Number of regions for adaptive clustering
    float delta_tolerance; // Initialization is Public
    float min_leaf_size; // no use
    float max_leaf_size; // no use
    float thresh_iou; // Intersection over Union threshold for clustering
    
    // L-shape fitting parameters
    float projection_range; // Projection range for L-shape fitting

    // Region of Interest (ROI) settings
    float roi_max_x; // Maximum x dimension for ROI
    float roi_max_y; // Maximum y dimension for ROI
    float roi_max_z; // Maximum z dimension for ROI
    float roi_min_x; // Minimum x dimension for ROI
    float roi_min_y; // Minimum y dimension for ROI
    float roi_min_z; // Minimum z dimension for ROI
    bool crop_ring_enabled; // Enable cropping by ring number
    int crop_ring; // Specific ring number to crop
    bool crop_intensity_enabled; // Enable cropping by intensity
    float crop_intensity; // Intensity threshold for cropping
    float crop_hd_map_radius; // Radius for HD map-based cropping

    // lidar
    ros::Time cur_stamp;
    ros::Time pre_stamp;
    double dt_l_c_;

    // imu
    GyrInt gyr_int_;
    double last_timestamp_imu;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    message_filters::Cache<sensor_msgs::Imu> imu_cache;
    sensor_msgs::ImuConstPtr last_imu_;
    Sophus::SE3d T_i_l;

    // patchworkpp
    boost::shared_ptr<PatchWorkpp<PointT>> PatchworkppGroundSeg;

    // average time check
    std::string package_path = ros::package::getPath("lidar_tracking") + "/time_log/cloud_segmentation/";
    std::string projection_time_log_path = package_path + "projection.txt";
    std::string convert_time_log_path = package_path + "convert.txt";
    std::string crop_time_log_path = package_path + "crop.txt";
    std::string crophdmap_time_log_path = package_path + "crophdmap.txt";
    std::string removalground_time_log_path = package_path + "removalground.txt";
    std::string undistortion_time_log_path = package_path + "undistortion.txt";
    std::string downsampling_time_log_path = package_path + "downsampling.txt";
    std::string clustering_time_log_path = package_path + "clustering.txt";
    std::string lshape_time_log_path = package_path + "lshape.txt";
    std::string average_time_log_path = package_path + "average.txt";

};

template<typename PointT> inline
void CloudSegmentation<PointT>::msgToPointCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg, pcl::PointCloud<PointT>& cloud)
{
    pcl::fromROSMsg(*cloud_msg, cloud);
    pre_stamp = cur_stamp;
    cur_stamp = cloud_msg->header.stamp;
    dt_l_c_ = cur_stamp.toSec() - pre_stamp.toSec();
}

template<typename PointT> inline
void CloudSegmentation<PointT>::imuUpdate(const sensor_msgs::Imu::ConstPtr &imu_msg)
{   
    imu_cache.add(imu_msg);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::projectPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- projectPointCloud" << std::endl;
        return;
    }

    cloudOut.clear();
    cloudOut.points.resize(V_SCAN * H_SCAN, PointT());

    std::vector<float> channelAngles = {14.985, 13.283, 11.758, 10.483, 9.836, 9.171, 8.496, 7.812, 7.462, 7.115, 6.767, 6.416, 6.064, 5.71,
                                        5.355, 4.998, 4.643, 4.282, 3.921, 3.558, 3.194, 2.829, 2.463, 2.095, 1.974, 1.854, 1.729, 1.609, 1.487, 1.362, 1.242, 1.12, 0.995, 0.875, 0.75,
                                        0.625, 0.5, 0.375, 0.25, 0.125, 0, -0.125, -0.25, -0.375, -0.5, -0.626, -0.751, -0.876, -1.001, -1.126, -1.251,
                                        -1.377, -1.502, -1.627, -1.751, -1.876, -2.001, -2.126, -2.251, -2.376, -2.501, -2.626, -2.751, -2.876, -3.001, -3.126, -3.251, -3.376, -3.501,
                                        -3.626, -3.751, -3.876, -4.001, -4.126, -4.25, -4.375, -4.501, -4.626, -4.751, -4.876, -5.001, -5.126, -5.252, -5.377, -5.502, -5.626, -5.752,
                                        -5.877, -6.002, -6.378, -6.754, -7.13, -7.507, -7.882, -8.257, -8.632, -9.003, -9.376, -9.749, -10.121, -10.493, -10.864, -11.234, -11.603, -11.975,
                                        -12.343, -12.709, -13.075, -13.439, -13.803, -14.164, -14.525, -14.879, -15.237, -15.593, -15.948, -16.299, -16.651, -17, -17.347, -17.701, -18.386,
                                        -19.063, -19.73, -20.376, -21.653, -23.044, -24.765};

    for (const auto& inPoint : cloudIn.points) {
        float verticalAngle = atan2(inPoint.z, sqrt(inPoint.x * inPoint.x + inPoint.y * inPoint.y)) * 180 / M_PI;

        // Find the closest channel angle
        auto it = std::lower_bound(channelAngles.begin(), channelAngles.end(), verticalAngle, std::greater<>());
        if (it == channelAngles.end()) continue;  // 각도가 모든 채널의 범위를 벗어난 경우

        size_t rowIdn = std::distance(channelAngles.begin(), it);

        float horizonAngle = atan2(inPoint.x, inPoint.y) * 180 / M_PI;
        size_t columnIdn = static_cast<size_t>((horizonAngle + 180.0) / 360.0 * H_SCAN);

        if (columnIdn >= H_SCAN) {
            columnIdn -= H_SCAN;
        }

        size_t index = columnIdn + rowIdn * H_SCAN;
        if (index < cloudOut.points.size()) {
            cloudOut.points[index] = inPoint;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(projection_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::convertPointCloudToImage(const pcl::PointCloud<PointT>& cloudIn, cv::Mat& imageOut, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- convertPointCloudToImage" << std::endl;
        return;
    }

    cv::Mat temp_image = cv::Mat::zeros(V_SCAN, H_SCAN, CV_8UC1);

    for (int i = 0; i < V_SCAN; ++i) {
        for (int j = 0; j < H_SCAN; ++j) {
            int index = i * H_SCAN + j;
            if (index < cloudIn.points.size() && !std::isnan(cloudIn.points[index].intensity)) {
                temp_image.at<uchar>(i, j) = static_cast<uchar>(cloudIn.points[index].intensity);
            }
        }
    }

    // Apply histogram equalization to enhance the image contrast
    cv::equalizeHist(temp_image, imageOut);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(convert_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::cropPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloud" << std::endl;
        return;
    }
    
    cloudOut.clear();
    cloudOut.reserve(cloudIn.size());
    
    for (const auto& point : cloudIn.points) {

        // ring filtering
        if (crop_ring_enabled && point.ring % crop_ring == 0) { continue; }
        // intensity filtering
        if (crop_intensity_enabled && point.intensity < crop_intensity) { continue; }
        
        // Car exclusion
        if (point.x >= -10.0 && point.x <= 2.0 &&
            point.y >= -0.8 && point.y <= 0.8) { continue; }

        // Rectangle
        if (point.x >= roi_min_x && point.x <= roi_max_x &&
            point.y >= roi_min_y && point.y <= roi_max_y &&
            point.z >= roi_min_z && point.z <= roi_max_z) {
            cloudOut.push_back(point);
        }

        // Degree
        // double min_angle_rad = -40 * M_PI / 180.0;
        // double max_angle_rad = 40 * M_PI / 180.0;
        // double angle = std::atan2(point.y, point.x);
        // if (angle >= min_angle_rad && angle <= max_angle_rad)
        // {
        //     cloudOut.push_back(point);
        // }

        // Circle
        // float distance = std::sqrt(point.x * point.x + point.y * point.y);
        // if (distance < R)
        // {
        //     cloudOut.push_back(point);
        // }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(crop_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::cropHDMapPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, 
                                                    tf2_ros::Buffer &tf_buffer, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloudHDMap" << std::endl;
        return;
    }

    cloudOut.clear();
    cloudOut.reserve(cloudIn.size());

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // input_stamp
    } catch (tf2::TransformException &ex) {
        // ROS_WARN("%s", ex.what());
        cloudOut = cloudIn;
        return;  
    }

    double ego_x = transformStamped.transform.translation.x;
    double ego_y = transformStamped.transform.translation.y;

    double min_dis = std::numeric_limits<double>::max();
    int min_idx = 0;
    int path_len = global_path.size();

    // Find the closest point on the global path to the ego vehicle's position
    for (int i = 0; i < path_len; ++i) {
        double dis = std::hypot(global_path[i].first - ego_x, global_path[i].second - ego_y);
        if (min_dis > dis) {
            min_dis = dis;
            min_idx = i;
        }
    }

    // Define the range of indices around the closest point
    int curr_index = min_idx;
    std::vector<int> indices;
    for (int k = 0; k < 180; ++k) {
        indices.push_back((curr_index + path_len + k - 80) % path_len);
    }

    // Transform and filter points based on proximity to the selected global path range
    // #pragma omp parallel for
    for (const auto& point : cloudIn.points) {
        geometry_msgs::Point geo_point;
        geo_point.x = point.x;
        geo_point.y = point.y;
        geo_point.z = point.z;

        geometry_msgs::Point transformed_point;
        tf2::doTransform(geo_point, transformed_point, transformStamped);

        for (int idx : indices) {
            double distance = std::hypot(global_path[idx].first - transformed_point.x, global_path[idx].second - transformed_point.y);
            if (distance <= crop_hd_map_radius) {
                // #pragma omp critical
                cloudOut.points.push_back(point);
                break;
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(crophdmap_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::removalGroundPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloud" << std::endl;
        return;
    }

    pcl::PointCloud<PointT> groundCloud;
    PatchworkppGroundSeg->estimate_ground(cloudIn, groundCloud, cloudOut, time_taken);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(removalground_time_log_path, time_taken);

}

template<typename PointT> inline
void CloudSegmentation<PointT>::undistortPointCloud(const pcl::PointCloud<PointT>& cloudIn, 
                                                    pcl::PointCloud<PointT>& cloudOut, double &time_taken)
{
    auto start_time = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- undistortPointCloud" << std::endl;
        cloudOut = cloudIn;
        return;
    }

    auto imu_data = imu_cache.getInterval(pre_stamp, cur_stamp);
    if (imu_data.empty()) {
        cloudOut = cloudIn;
        std::cerr << "Input imu is empty! <- undistortPointCloud" << std::endl;
        return;
    }

    const sensor_msgs::Imu::ConstPtr& first_imu = imu_data.front();
    double timestamp = first_imu->header.stamp.toSec();
    
    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("IMU loop back detected, resetting integrator");
        gyr_int_.Reset(timestamp, first_imu);
    }
    last_timestamp_imu = timestamp;  // 최신 IMU 타임스탬프 업데이트

    // 타임스탬프 범위 내의 IMU 데이터를 통합
    gyr_int_.Reset(pre_stamp.toSec(), first_imu);  // Reset integrator with the first IMU in the time range
    for (const auto& imu : imu_data) {
        gyr_int_.Integrate(imu);  // 각 IMU 데이터를 통합하여 회전값 계산
    }

    // IMU와 LiDAR 간의 변환 행렬 T_i_l을 이용해 보정할 최종 회전 및 변환 행렬 생성
    Sophus::SE3d T_l_c(gyr_int_.GetRot(), Eigen::Vector3d::Zero());  // 현재 IMU 회전만 고려
    Sophus::SE3d T_l_be = T_i_l.inverse() * T_l_c * T_i_l;  // LiDAR 좌표계에서 IMU 좌표계로 변환 적용

    // 회전과 변환 정보 추출
    const Eigen::Vector3d &tbe = T_l_be.translation();
    Eigen::Vector3d rso3_be = T_l_be.so3().log();

    // 각 포인트에 대해 회전 및 변환 적용
    cloudOut.clear();
    cloudOut.reserve(cloudIn.size());

    for (const auto &pt : cloudIn.points) {
        // 포인트의 시간 차이 계산
        double dt_bi = pt.timestamp - pre_stamp.toSec();
        if (dt_bi == 0) continue;

        double ratio_bi = dt_bi / dt_l_c_;
        double ratio_ie = 1 - ratio_bi;

        // IMU 회전을 LiDAR 프레임으로 보정
        Eigen::Vector3d rso3_ie = ratio_ie * rso3_be;
        Sophus::SO3d Rie = Sophus::SO3d::exp(rso3_ie);

        // 보정된 회전과 변환 적용: P_compensate = R_ei * Pi + t_ei
        Eigen::Vector3d tie = ratio_ie * tbe;
        Eigen::Vector3d pt_vec(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_compensated = Rie.inverse() * (pt_vec - tie);

        // 보정된 포인트 업데이트
        PointT new_point = pt;
        new_point.x = pt_compensated.x();
        new_point.y = pt_compensated.y();
        new_point.z = pt_compensated.z();
        cloudOut.points.push_back(new_point);
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(undistortion_time_log_path, time_taken);
}

// CUDA-PointPillars
template<typename PointT> inline
void CloudSegmentation<PointT>::pcl2FloatArray(const pcl::PointCloud<PointT>& cloudIn, std::vector<float>& arrayOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- pcl2FloatArray" << std::endl;
        return;
    }

    size_t num_points = cloudIn.size();
    arrayOut.resize(num_points * 4);  // 각 포인트마다 x, y, z, intensity

    for (size_t i = 0; i < num_points; ++i) {
        arrayOut[i * 4 + 0] = cloudIn.points[i].x;
        arrayOut[i * 4 + 1] = cloudIn.points[i].y;
        arrayOut[i * 4 + 2] = cloudIn.points[i].z;
        arrayOut[i * 4 + 3] = cloudIn.points[i].intensity / 255.0f; // intensity 정규화
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

template<typename PointT> inline
void CloudSegmentation<PointT>::downsamplingPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<ClusterPointT>& cloudOut, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- downsamplingPointCloud" << std::endl;
        return;
    }

    cloudOut.clear();
    cloudOut.reserve(cloudIn.size());

    pcl::PointCloud<ClusterPointT> tempCloud;
    pcl::copyPointCloud(cloudIn, tempCloud); // copyPointCloud 사용하여 타입 변환 및 데이터 복사
    pcl::VoxelGrid<ClusterPointT> voxel_grid_filter;
    voxel_grid_filter.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(tempCloud)); // 입력 클라우드 설정
    voxel_grid_filter.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z); // Voxel 크기 설정
    voxel_grid_filter.filter(cloudOut); // 필터링 수행
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(downsampling_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::adaptiveClustering(const pcl::PointCloud<ClusterPointT>& cloudIn, 
                                                std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- adaptiveClustering" << std::endl;
        return;
    }

    outputClusters.clear();
    outputClusters.reserve(cloudIn.size());

    // Divide the point cloud into nested circular regions
    std::vector<float> regions(max_region_distance, max_region_distance / number_region); // Example: Fill regions with a distance increment of 15m each
    std::vector<std::vector<int>> indices_array(max_region_distance);

    for (int i = 0; i < cloudIn.size(); i++) {
        float distance = cloudIn.points[i].x * cloudIn.points[i].x + cloudIn.points[i].y * cloudIn.points[i].y;
        float range = 0.0;
        for (int j = 0; j < max_region_distance; j++) {
            if (distance > range * range && distance <= (range + regions[j]) * (range + regions[j]))
            {
                indices_array[j].push_back(i);
                break;
            }
            range += regions[j];
        }
    }


    // Euclidean clustering for each region
    for (int i = 0; i < max_region_distance; i++) {
        if (indices_array[i].empty()) continue;

        pcl::PointCloud<ClusterPointT> cloudSegment;
        for (int index : indices_array[i]) {
            cloudSegment.points.push_back(cloudIn.points[index]);
        }

        pcl::search::KdTree<ClusterPointT> tree;
        tree.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<ClusterPointT> ec;
        ec.setClusterTolerance(min_tolerance + delta_tolerance * i); // Increment tolerance for farther regions
        ec.setMinClusterSize(adaptive_min_size);
        ec.setMaxClusterSize(adaptive_max_size);
        ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<ClusterPointT>>(tree));
        ec.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));
        ec.extract(cluster_indices);

        for (auto& indices : cluster_indices) {
            pcl::PointCloud<ClusterPointT> cluster;
            for (int idx : indices.indices) {
                cluster.points.push_back(cloudSegment.points[idx]);
            }
            cluster.width = cluster.size();
            cluster.height = 1;
            cluster.is_dense = true;
            
            // Size filtering
            ClusterPointT minPt, maxPt;
            pcl::getMinMax3D(cluster, minPt, maxPt);
            double clusterSizeX = maxPt.x - minPt.x;
            double clusterSizeY = maxPt.y - minPt.y;
            double clusterSizeZ = maxPt.z - minPt.z;

            if (clusterSizeX > filter_min_size_x && clusterSizeX < filter_max_size_x &&
                clusterSizeY > filter_min_size_y && clusterSizeY < filter_max_size_y &&
                clusterSizeZ > filter_min_size_z && clusterSizeZ < filter_max_size_z) {
                outputClusters.push_back(cluster);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    time_taken = elapsed.count();
    saveTimeToFile(clustering_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::adaptiveVoxelClustering(const pcl::PointCloud<PointT>& cloudIn, 
                                                     std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, 
                                                     double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- adaptiveVoxelClustering" << std::endl;
        return;
    }

    outputClusters.clear();
    outputClusters.reserve(cloudIn.size());

    // Convert the input cloud to ClusterPointT type
    pcl::PointCloud<ClusterPointT> xyzCloud;
    pcl::copyPointCloud(cloudIn, xyzCloud);

    // 구간별 거리 단계 설정
    std::vector<float> regions(number_region, max_region_distance / number_region); // 10m씩 증가하는 구간
    std::vector<std::vector<int>> indices_array(number_region + 1); // 추가적으로 100m 이후를 위한 구간 추가

    // 포인트 클라우드를 구간별로 분류
    for (int i = 0; i < xyzCloud.size(); i++) {
        float distance = std::sqrt(xyzCloud.points[i].x * xyzCloud.points[i].x + xyzCloud.points[i].y * xyzCloud.points[i].y);
        if (distance > max_region_distance) {
            // 100m 이후의 포인트는 마지막 구간에 배치
            indices_array[number_region].push_back(i);
        } else {

            // 100m 이후의 포인트는 마지막 구간에 배치
            if (distance > max_region_distance) {
                indices_array[number_region].push_back(i);
            } else {
                // 구간을 계산하여 인덱스 선택
                int region_index = static_cast<int>(distance / (max_region_distance / number_region));
                indices_array[region_index].push_back(i);
            }
        }
    }

    // Iterate over each region and apply voxel grid filtering and clustering 
    for (int i = 0; i <= number_region; i++) {
        if (indices_array[i].empty()) continue;

        pcl::PointCloud<ClusterPointT> cloudSegment;
        for (int index : indices_array[i]) {
            cloudSegment.points.push_back(xyzCloud.points[index]);
        }

        if (i != number_region) {
            // 100m 이전의 구간 처리 (복셀화 적용)
            pcl::VoxelGrid<ClusterPointT> voxel_grid_filter;
            float leaf_size = max_leaf_size - (i * (max_leaf_size - min_leaf_size) / (number_region - 1)); // 구간별 leaf_size 계산
            voxel_grid_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_grid_filter.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));

            pcl::PointCloud<ClusterPointT> downsampledCloud;
            voxel_grid_filter.filter(downsampledCloud);
            cloudSegment = downsampledCloud; // 다운샘플링된 클라우드로 업데이트
        }

        // Perform Euclidean clustering
        pcl::search::KdTree<ClusterPointT> tree;
        tree.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<ClusterPointT> ec;
        float tolerance = (i == number_region) ? max_tolerance : min_tolerance + (i * (max_tolerance - min_tolerance) / (number_region - 1)); // 구간별 tolerance 계산
        ec.setClusterTolerance(tolerance);
        ec.setMinClusterSize(adaptive_min_size);  // 원래 코드에 있던 adaptive_min_size 사용
        ec.setMaxClusterSize(adaptive_max_size);  // 원래 코드에 있던 adaptive_max_size 사용
        ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<ClusterPointT>>(tree));
        ec.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));
        ec.extract(cluster_indices);

        // Cluster size filtering and output storage
        for (auto& indices : cluster_indices) {
            pcl::PointCloud<ClusterPointT> cluster;
            for (int idx : indices.indices) {
                cluster.points.push_back(cloudSegment.points[idx]);
            }
            cluster.width = cluster.size();
            cluster.height = 1;
            cluster.is_dense = true;

            outputClusters.push_back(cluster);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    time_taken = elapsed.count();
    saveTimeToFile(clustering_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::fittingLShape(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters, 
                                           jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (inputClusters.empty()) {
        std::cerr << "Input clusters is empty! <- fittingLShape" << std::endl;
        return;
    }

    output_bbox_array.boxes.clear();

    for (const pcl::PointCloud<ClusterPointT>& cluster : inputClusters) {
        ClusterPointT minPoint, maxPoint;
        pcl::getMinMax3D(cluster, minPoint, maxPoint);

        // rectangle
        LShapedFIT lshaped;
        std::vector<cv::Point2f> points = pcl2Point2f(cluster, projection_range); // Convert 3D cluster to BEV projection
        cv::RotatedRect rr = lshaped.FitBox(&points);
        std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
        double yaw = (rr.angle + 90) * M_PI / 180.0; // Convert degrees to radians

        if (rr.center.x == 0.0) { continue; } // Skip clusters at the origin

        // Create jsk_recognition_msgs::BoundingBox
        jsk_recognition_msgs::BoundingBox bbox;
        bbox.header.stamp = cur_stamp;
        bbox.header.frame_id = lidar_frame;
        bbox.pose.position.x = rr.center.x;
        bbox.pose.position.y = rr.center.y;
        bbox.pose.position.z = (minPoint.z + maxPoint.z) / 2.0; // Set z-axis height to be halfway between min and max
        bbox.dimensions.x = rr.size.height;
        bbox.dimensions.y = rr.size.width;
        bbox.dimensions.z = maxPoint.z - minPoint.z;
        bbox.pose.orientation.z = std::sin(yaw / 2.0);
        bbox.pose.orientation.w = std::cos(yaw / 2.0);

        if (bbox.dimensions.x < filter_min_size_x || bbox.dimensions.x > filter_max_size_x ||
            bbox.dimensions.y < filter_min_size_y || bbox.dimensions.y > filter_max_size_y ||
            bbox.dimensions.z < filter_min_size_z || bbox.dimensions.z > filter_max_size_z) {
            continue;
        }

        output_bbox_array.boxes.push_back(bbox);
    }

    for (int i = 0; i < output_bbox_array.boxes.size(); ++i) {
        for (int j = i + 1; j < output_bbox_array.boxes.size();) {
            double overlap = getBBoxOverlap(output_bbox_array.boxes[j], output_bbox_array.boxes[i]);
            if (overlap > thresh_iou) {
                auto& box_i = output_bbox_array.boxes[i];
                auto& box_j = output_bbox_array.boxes[j];

                double volume_i = box_i.dimensions.x * box_i.dimensions.y * box_i.dimensions.z;
                double volume_j = box_j.dimensions.x * box_j.dimensions.y * box_j.dimensions.z;

                if (volume_i >= volume_j) {
                    output_bbox_array.boxes.erase(output_bbox_array.boxes.begin() + j);
                } else {
                    output_bbox_array.boxes.erase(output_bbox_array.boxes.begin() + i);
                    --i;
                    break;
                }
            } else {
                ++j;
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(lshape_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::averageTime()
{
    std::ofstream file(average_time_log_path, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << average_time_log_path << std::endl;
        return;
    }

    file << "projection : " << calculateAverageTime(projection_time_log_path) << "\n";
    file << "converstion : " << calculateAverageTime(convert_time_log_path) << "\n";
    file << "crop : " << calculateAverageTime(crop_time_log_path) << "\n";
    file << "ground removal : " << calculateAverageTime(removalground_time_log_path) << "\n";
    file << "undistortion : " << calculateAverageTime(undistortion_time_log_path) << "\n";
    file << "downsampling : " << calculateAverageTime(downsampling_time_log_path) << "\n";
    file << "clustering : " << calculateAverageTime(clustering_time_log_path) << "\n";
    file << "Lshape fitting : " << calculateAverageTime(lshape_time_log_path) << "\n";

    file.close();
}

