#include "utils.hpp"
#include "patchworkpp/patchworkpp.hpp"

using ClusterPointT = pcl::PointXYZI;

template <typename PointT>
class CloudSegmentation {
public:
    CloudSegmentation() {};

    CloudSegmentation(ros::NodeHandle& nh) : nh_(nh) {
        nh_.getParam("Public/lidar_frame", lidar_frame);
        nh_.getParam("Public/target_frame", target_frame);
        nh_.getParam("Public/world_frame", world_frame);
        nh_.getParam("Cloud_Segmentation/lidar_settings/V_SCAN", V_SCAN);
        nh_.getParam("Cloud_Segmentation/lidar_settings/H_SCAN", H_SCAN);
        nh_.getParam("Cloud_Segmentation/lidar_settings/resolution_x", resolution_x);
        nh_.getParam("Cloud_Segmentation/lidar_settings/resolution_y", resolution_y);
        nh_.getParam("Cloud_Segmentation/lidar_settings/ang_bottom", ang_bottom);
        nh_.getParam("Cloud_Segmentation/crop/max/x", roi_max_x);
        nh_.getParam("Cloud_Segmentation/crop/max/y", roi_max_y);
        nh_.getParam("Cloud_Segmentation/crop/max/z", roi_max_z);
        nh_.getParam("Cloud_Segmentation/crop/min/x", roi_min_x);
        nh_.getParam("Cloud_Segmentation/crop/min/y", roi_min_y);
        nh_.getParam("Cloud_Segmentation/crop/min/z", roi_min_z);
        nh_.getParam("Cloud_Segmentation/ground_removal/fp_distance", fp_distance);
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
        nh_.getParam("Cloud_Segmentation/crop/crop_ring/enabled", crop_ring_enabled);
        nh_.getParam("Cloud_Segmentation/crop/crop_ring/ring", crop_ring);
        nh_.getParam("Cloud_Segmentation/crop/crop_intensity/enabled", crop_intensity_enabled);
        nh_.getParam("Cloud_Segmentation/crop/crop_intensity/intensity", crop_intensity);
        nh_.getParam("Cloud_Segmentation/crop/crop_hd_map/radius", crop_hd_map_radius);

        global_path = map_reader(map.c_str());

        // lidar
        dt_l_c_ = 0.1;
        cur_stamp = ros::Time(0);
        pre_stamp = ros::Time(0);

        // imu
        imu_cache.setCacheSize(1000);
        last_timestamp_imu = -1;
        // Eigen::Quaterniond q(1, 0, 0, 0);
        Eigen::Quaterniond q(std::sqrt(2)/2, 0, 0, std::sqrt(2)/2);
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
    }

    void msgToPointCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, pcl::PointCloud<PointT>& cloud);
    void updateImu(const sensor_msgs::Imu::ConstPtr &imu_msg);
    void projectPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    void convertPointCloudToImage(const pcl::PointCloud<PointT>& cloudIn, cv::Mat &imageOut, double &time_taken);
    void cropPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    void cropHDMapPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, 
                            tf2_ros::Buffer &tf_buffer, double &time_taken);
    void removalGroundPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, 
                                pcl::PointCloud<PointT>& ground, double &time_taken);
    void undistortPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, double &time_taken);
    
    // CUDA-PointPillars
    void pcl2FloatArray(const pcl::PointCloud<PointT>& cloudIn, std::vector<float>& arrayOut, double &time_taken);

    void downsamplingPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<ClusterPointT>& cloudOut, double &time_taken);
    void adaptiveClustering(const pcl::PointCloud<PointT>& cloudIn, std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double &time_taken);
    void voxelClustering(const pcl::PointCloud<PointT>& cloudIn, std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double& time_taken);
    void fittingLShape(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters,
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    void fittingPCA(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters, 
                    jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken);

    void adaptiveVoxelClustering(const pcl::PointCloud<PointT>& cloudIn, std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double& time_taken);

private:
    ros::NodeHandle nh_;
    std::string map;
    std::string lidar_frame;
    std::string target_frame;
    std::string world_frame;
    std::vector<std::pair<float, float>> global_path;
    int V_SCAN; // Vertical scan lines
    int H_SCAN; // Horizontal scan points per line
    float resolution_x; // Angular resolution in x direction (degrees)
    float resolution_y; // Angular resolution in y direction (degrees)
    int ang_bottom; // Bottom angle (degrees)

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

    // Ground Removal parameters
    float fp_distance;

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

};

template<typename PointT> inline
void CloudSegmentation<PointT>::msgToPointCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, pcl::PointCloud<PointT>& cloud)
{
    pcl::fromROSMsg(*cloud_msg, cloud);
    pre_stamp = cur_stamp;
    cur_stamp = cloud_msg->header.stamp;
    dt_l_c_ = cur_stamp.toSec() - pre_stamp.toSec();
}

template<typename PointT> inline
void CloudSegmentation<PointT>::updateImu(const sensor_msgs::Imu::ConstPtr &imu_msg)
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
    cloudOut.points.resize(V_SCAN * H_SCAN);

    // Pandar64
    std::vector<float> channelAngles = {14.882, 11.032, 8.059, 5.057, 3.04, 2.028, 1.86, 1.688, 1.522, 1.351, 1.184, 1.013,
                                        0.846, 0.675, 0.508, 0.337, 0.169, 0.000, -0.169, -0.337, -0.508, -0.675, -0.845, -1.013,
                                        -1.184, -1.351, -1.522, -1.688, -1.86, -2.028, -2.198, -2.365, -2.536, -2.7, -2.873, -3.04,
                                        -3.21, -3.375, -3.548, -3.712, -3.884, -4.05, -4.221, -4.385, -4.558, -4.72, -4.892, -5.057,
                                        -5.229, -5.391, -5.565, -5.731, -5.898, -6.061, -7.063, -8.059, -9.06, -9.885, -11.032, -12.006,
                                        -12.974, -13.93, -18.889, -24.897};

    for (const auto& inPoint : cloudIn.points)
    {
        PointT outPoint = inPoint;

        // 수직 각도를 계산하여 해당 레이저 채널을 찾음
        float verticalAngle = atan2(outPoint.z, sqrt(outPoint.x * outPoint.x + outPoint.y * outPoint.y)) * 180 / M_PI;

        // 가장 가까운 수직 각도 인덱스를 찾음
        auto it = std::min_element(channelAngles.begin(), channelAngles.end(),
                                   [verticalAngle](float a, float b) {
                                       return fabs(a - verticalAngle) < fabs(b - verticalAngle);
                                   });
        if (it == channelAngles.end()) continue;

        size_t rowIdn = std::distance(channelAngles.begin(), it);  // 각도에 맞는 수직 인덱스(row)

        if (rowIdn < 0 || rowIdn >= V_SCAN)
            continue;

        // 수평 각도 계산 및 인덱스
        float horizonAngle = atan2(outPoint.x, outPoint.y) * 180 / M_PI;
        size_t columnIdn = static_cast<size_t>(-round((horizonAngle - 90.0) / resolution_x) + H_SCAN / 2);

        if (columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if (columnIdn < 0 || columnIdn >= H_SCAN)
            continue;

        size_t index = columnIdn + rowIdn * H_SCAN;

        if (index < cloudOut.points.size()) {
            cloudOut.points[index] = outPoint;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(projection_time_log_path, time_taken);
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
    // saveTimeToFile(convert_time_log_path, time_taken);
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
        if (point.x >= -2.0 && point.x <= 2.0 &&
            point.y >= -0.9 && point.y <= 0.9) { continue; }

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
    // saveTimeToFile(crop_time_log_path, time_taken);
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
    // saveTimeToFile(crophdmap_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::removalGroundPointCloud(const pcl::PointCloud<PointT>& cloudIn, pcl::PointCloud<PointT>& cloudOut, 
                                                        pcl::PointCloud<PointT>& ground, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloud" << std::endl;
        return;
    }

    pcl::PointCloud<PointT> groundCloud, nongroundCloud;
    PatchworkppGroundSeg->estimate_ground(cloudIn, groundCloud, nongroundCloud, time_taken);

    // 근거리 지면 오인식 필터링
    for (const auto& pt : groundCloud.points) {
        double range = sqrt(pt.x * pt.x + pt.y * pt.y);

        if (range <= fp_distance && pt.z > roi_min_z + 0.3) {
            nongroundCloud.push_back(pt);
        }
    }

    cloudOut = nongroundCloud;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    // saveTimeToFile(removalground_time_log_path, time_taken);
}

// hesai
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
    // saveTimeToFile(undistortion_time_log_path, time_taken);
}

// OpenPCDet
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
        arrayOut[i * 4 + 3] = cloudIn.points[i].intensity / 255.0f;
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
    // saveTimeToFile(downsampling_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::adaptiveClustering(const pcl::PointCloud<PointT>& cloudIn, 
                                                std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- adaptiveClustering" << std::endl;
        return;
    }

    pcl::PointCloud<ClusterPointT> tempCloud;
    pcl::copyPointCloud(cloudIn, tempCloud);

    outputClusters.clear();
    outputClusters.reserve(tempCloud.size());

    // Divide the point cloud into nested circular regions
    std::vector<float> regions(max_region_distance, max_region_distance / number_region); // Example: Fill regions with a distance increment of 15m each
    std::vector<std::vector<int>> indices_array(max_region_distance);

    for (int i = 0; i < tempCloud.size(); i++) {
        float distance = tempCloud.points[i].x * tempCloud.points[i].x + tempCloud.points[i].y * tempCloud.points[i].y;
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
            cloudSegment.points.push_back(tempCloud.points[index]);
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
    // saveTimeToFile(clustering_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::adaptiveVoxelClustering(const pcl::PointCloud<PointT>& cloudIn, 
                                                     std::vector<pcl::PointCloud<ClusterPointT>>& outputClusters, 
                                                     double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn.points.empty()) {
        std::cerr << "Input cloud is empty! <- adaptiveVoxelClustering" << std::endl;
        return;
    }

    outputClusters.clear();
    outputClusters.reserve(cloudIn.size());

    pcl::PointCloud<ClusterPointT> xyzCloud;
    pcl::copyPointCloud(cloudIn, xyzCloud);

    std::vector<float> regions(number_region, max_region_distance / number_region); 
    std::vector<std::vector<int>> indices_array(number_region + 1); 

    for (int i = 0; i < xyzCloud.size(); i++) {
        float distance = std::sqrt(xyzCloud.points[i].x * xyzCloud.points[i].x + xyzCloud.points[i].y * xyzCloud.points[i].y);
        int region_index = distance > max_region_distance ? number_region : static_cast<int>(distance / (max_region_distance / number_region));
        indices_array[region_index].push_back(i);
    }

    // double total_downsampling_time = 0.0;
    // double total_clustering_time = 0.0;

    for (int i = 0; i <= number_region; i++) {
        if (indices_array[i].empty()) continue;

        pcl::PointCloud<ClusterPointT> cloudSegment;
        for (int index : indices_array[i]) {
            cloudSegment.points.push_back(xyzCloud.points[index]);
        }

        // 복셀화 적용
        if (i != number_region) {
            // auto downsample_start = std::chrono::steady_clock::now();
            
            pcl::VoxelGrid<ClusterPointT> voxel_grid_filter;
            float leaf_size = max_leaf_size - (i * (max_leaf_size - min_leaf_size) / (number_region - 1)); 
            voxel_grid_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_grid_filter.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));

            pcl::PointCloud<ClusterPointT> downsampledCloud;
            voxel_grid_filter.filter(downsampledCloud);
            cloudSegment = downsampledCloud; 
            
            // auto downsample_end = std::chrono::steady_clock::now();
            // total_downsampling_time += std::chrono::duration<double>(downsample_end - downsample_start).count();
        }
        
        // 클러스터링 수행
        // auto clustering_start = std::chrono::steady_clock::now();

        pcl::search::KdTree<ClusterPointT> tree;
        tree.setInputCloud(boost::make_shared<pcl::PointCloud<ClusterPointT>>(cloudSegment));

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<ClusterPointT> ec;
        float tolerance = (i == number_region) ? max_tolerance : min_tolerance + (i * (max_tolerance - min_tolerance) / (number_region - 1)); 
        ec.setClusterTolerance(tolerance);
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

            outputClusters.push_back(cluster);
        }

        // auto clustering_end = std::chrono::steady_clock::now();
        // total_clustering_time += std::chrono::duration<double>(clustering_end - clustering_start).count();
    }

    // saveTimeToFile(downsampling_time_log_path, total_downsampling_time);
    // saveTimeToFile(clustering_time_log_path, total_clustering_time);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    time_taken = elapsed.count();    
}

template<typename PointT> inline
void CloudSegmentation<PointT>::fittingLShape(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters, 
                                           jsk_recognition_msgs::BoundingBoxArray& output_bbox_array, double& time_taken) 
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
    // saveTimeToFile(lshape_time_log_path, time_taken);
}

template<typename PointT> inline
void CloudSegmentation<PointT>::fittingPCA(const std::vector<pcl::PointCloud<ClusterPointT>>& inputClusters, 
                                           jsk_recognition_msgs::BoundingBoxArray& output_bbox_array, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    if (inputClusters.empty()) {
        std::cerr << "Input clusters is empty! <- fittingPCA" << std::endl;
        return;
    }

    for (const pcl::PointCloud<ClusterPointT>& cluster : inputClusters)
    {
        if (cluster.points.empty())
            continue;

        // PCA를 위한 객체 생성
        pcl::PCA<ClusterPointT> pca;
        pca.setInputCloud(cluster.makeShared());

        // 중심 좌표 계산
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(cluster, centroid);

        // 고유값 및 고유벡터 계산
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

        // 변환 행렬 생성
        Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
        transform.block<3,3>(0,0) = eigen_vectors.transpose();
        transform.block<3,1>(0,3) = -1.0f * (eigen_vectors.transpose() * centroid.head<3>());

        // 점군 변환
        pcl::PointCloud<ClusterPointT> transformedCloud;
        pcl::transformPointCloud(cluster, transformedCloud, transform);

        // 변환된 점군의 최소 및 최대 좌표 계산
        ClusterPointT minPoint, maxPoint;
        pcl::getMinMax3D(transformedCloud, minPoint, maxPoint);

        // 바운딩 박스의 중심 및 크기 계산
        Eigen::Vector3f mean_diag = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

        Eigen::Quaternionf bbox_quaternion(eigen_vectors);
        Eigen::Vector3f bbox_transform = eigen_vectors * mean_diag + centroid.head<3>();

        // 바운딩 박스 생성
        jsk_recognition_msgs::BoundingBox bbox;
        bbox.header.stamp = cur_stamp;
        bbox.header.frame_id = lidar_frame;
        bbox.pose.position.x = bbox_transform.x();
        bbox.pose.position.y = bbox_transform.y();
        bbox.pose.position.z = bbox_transform.z();
        bbox.dimensions.x = maxPoint.x - minPoint.x;
        bbox.dimensions.y = maxPoint.y - minPoint.y;
        bbox.dimensions.z = maxPoint.z - minPoint.z;
        // bbox.pose.orientation.x = bbox_quaternion.x();
        // bbox.pose.orientation.y = bbox_quaternion.y();
        bbox.pose.orientation.x = 0;
        bbox.pose.orientation.y = 0;
        bbox.pose.orientation.z = bbox_quaternion.z();
        bbox.pose.orientation.w = bbox_quaternion.w();

        // 크기 필터 적용 (fittingLShape와 동일하게)
        if (bbox.dimensions.x < filter_min_size_x || bbox.dimensions.x > filter_max_size_x ||
            bbox.dimensions.y < filter_min_size_y || bbox.dimensions.y > filter_max_size_y ||
            bbox.dimensions.z < filter_min_size_z || bbox.dimensions.z > filter_max_size_z) {
            continue;
        }

        output_bbox_array.boxes.push_back(bbox);
    }

    // 겹치는 바운딩 박스 제거 (fittingLShape와 동일한 방식으로)
    for (size_t i = 0; i < output_bbox_array.boxes.size(); ++i) {
        for (size_t j = i + 1; j < output_bbox_array.boxes.size();) {
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
    // saveTimeToFile(lshape_time_log_path, time_taken);
}