# patchwork-plusplus-ros

This is ROS package of LiDAR Tracking

<p align="center"><img src=pictures/patchwork++.gif alt="animated" /></p>


## :open_file_folder: What's in this repository

* ROS based LiDAR Tracking source code
* Demo launch file ([demo.launch][launchlink]) with sample rosbag file. You can execute simply!

[launchlink]: https://github.com/inha-cvl/LiDAR-Tracking/tree/main/launch

## :package: Prerequisite packages
You may need to install ROS, PCL, Eigen, ...

## :gear: How to build Patchwork++
To build LiDAR-Tracking, you can follow below codes.

```bash
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws
$ catkin_make
```

## :runner: To run the demo codes

```bash
# Start Patchwork++
$ roslaunch lidar_tracking demo.launch
# Start the bag file
$ roscore
$ rosparam set /use_sim_time true
$ rosbag play kiapi.bag
```

## :pushpin: TODO List
- [ ] Add efficient adaptive clustering 
- [ ] Add tracking source code
- [ ] Add Refernce Patchwork, undistortion etc..

## Citation
If you use our codes, please cite our [paper]


[paper]: --



## :postbox: Contact
If you have any question, don't be hesitate let us know!

* [Gyuseok Lee][link] :envelope: (rbtjr98@inha.edu)

[link]: https://github.com/Lee-Gyu-Seok

