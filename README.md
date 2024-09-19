# LiDAR Tracking

This is ROS package of LiDAR Tracking

<p align="center"><img src=pictures/demo.gif alt="animated" /></p>


## :open_file_folder: What's in this repository

* ROS Noetic based LiDAR Tracking source code
* Demo launch file ([demo.launch][launchlink])

[launchlink]: https://github.com/inha-cvl/LiDAR-Tracking/tree/main/launch

## :package: Prerequisite packages
You need to install ROS, PCL, Sophus, Eigen, JSK, Json...

## :gear: How to build LiDAR-Tracking

```bash
$ mkdir -p ~/catkin_ws/src
$ git clone https://github.com/inha-cvl/LiDAR-Tracking.git
$ cd ~/catkin_ws
$ catkin_make
```

## :runner: To run the demo

```bash
# Start Tracking with bag file
$ roscore
$ roslaunch lidar_tracking demo.launch
$ rosparam set /use_sim_time true
$ rosbag play kiapi.bag --clock
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

