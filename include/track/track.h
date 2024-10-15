#ifndef TRACK_H
#define TRACK_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <angles/angles.h>
#include <tf/tf.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <visualization_msgs/MarkerArray.h>

#include "track/HungarianAlg.h"

struct trackingStruct
{
	unsigned int id;
	unsigned int age;
	unsigned int cls;
	float score;
	unsigned cntTotalVisible;
	unsigned cntConsecutiveInvisible;

	jsk_recognition_msgs::BoundingBox cur_bbox;
	jsk_recognition_msgs::BoundingBox pre_bbox;

	float vx, vy, v, ax, ay;

	std::deque<float> vx_deque, vy_deque, v_deque;
	std::deque<float> orientation_deque;

	cv::KalmanFilter kf;

	double lastTime;
};

class Track
{
private:
	// Setting parameters
	int stateVariableDim;
	int stateMeasureDim;
	unsigned int nextID;
	unsigned int m_thres_invisibleCnt;
	int n_velocity_deque;
	int n_orientation_deque;
	float thr_velocity;
	float thr_orientation;


	cv::Mat m_matTransition;
	cv::Mat m_matMeasurement;

	cv::Mat m_matProcessNoiseCov;
	cv::Mat m_matMeasureNoiseCov;

	float m_thres_associationCost;

	float dt = 0.05;
	
	// Global variables
	vector<trackingStruct> vecTracks;
	vector<pair<int, int>> vecAssignments;
	vector<int> vecUnassignedTracks;
	vector<int> vecUnssignedDetections;

public:
	//constructor
	Track();
	//deconstructor
	~Track();
	
	void setParams(int invisibleCnt,
				   int number_velocity_deque,
				   int number_orientation_deque,
				   float thresh_velocity,
				   float thresh_orientation
	)
	{
		m_thres_invisibleCnt = invisibleCnt;
		n_velocity_deque = number_velocity_deque;
		n_orientation_deque = number_orientation_deque;
		thr_velocity = thresh_velocity;
		thr_orientation = thresh_orientation;
	};

	void velocity_push_back(std::deque<float> &deque, float v);
	void orientation_push_back(std::deque<float> &deque, float o);
	float getVectorScale(float v1, float v2);
	double getBBoxRatio(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2);
	double getBBoxDistance(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2);
	visualization_msgs::Marker get_text_msg(struct trackingStruct &track, int i);
	void predictNewLocationOfTracks(const ros::Time &currentTime);
	void assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray);
	void assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray);
	// void assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray, const geometry_msgs::PoseStamped &enu_pose);
	void unassignedTracksUpdate();
	void deleteLostTracks();
	void createNewTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxMarkerArray);
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> displayTrack();
};

#endif
