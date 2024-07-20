add_library(track_lib
            HungarianAlg.cpp
            track.cpp
            lshaped_fitting.cpp
            data_process.cpp
            gyr_int.cpp
            depth_clustering.cpp)
target_link_libraries(track_lib ${catkin_LIBRARIES} ${PCL_LIBRARIES})
