
# LiDAR Tracking

This is ROS package of LiDAR Tracking

<p align="center"><img src=pictures/integration.gif alt="animated" /></p>

## :open_file_folder: What's in this repository

* ROS Noetic based LiDAR Tracking source code
* Demo youtube link ([youtube][youtubelink])

[youtubelink]: https://www.youtube.com/watch?v=YP5AAO_Eq5Y

## :package: Prerequisite packages
You need to install ROS, PCL, Sophus, Glog, Eigen, JSK, Json...

## :gear: How to build LiDAR-Tracking

```bash
$ mkdir -p ~/catkin_ws/src
$ git clone https://github.com/inha-cvl/LiDAR-Tracking.git
$ cd ~/catkin_ws
$ catkin_make
```

## :runner: To run the demo

* Download bag ([bagfile][onedrivelink])

[onedrivelink]: https://1drv.ms/u/s!At4eTVNRwillgdsABY-z8AVEqmvoxg?e=PCqjYE

```bash
# Start Tracking with bag file
$ roscore
$ rosparam set /use_sim_time true
$ roslaunch lidar_tracking tracking.launch
$ roslaunch openpcdet 3d_object_detector.launch # Deep Learning-based bounding box publishing needed
$ roslaunch lidar_tracking integration.launch
$ rosbag play songdo.bag --clock
```

## :pushpin: References
- https://github.com/url-kaist/patchwork-plusplus-ros
- https://github.com/SmallMunich/L-Shape-Fitting
- https://github.com/yzrobot/adaptive_clustering.git

## Citation
If you use our codes, please cite our [paper]


[paper]: --


## :postbox: Contact
If you have any question, don't be hesitate let us know!

* [Gyuseok Lee][link] :envelope: (rbtjr98@inha.edu)

[link]: https://github.com/Lee-Gyu-Seok
