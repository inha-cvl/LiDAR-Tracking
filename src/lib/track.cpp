#include "track/track.h"
#include <cmath>

using namespace cv;

//constructor
Track::Track()
{
	stateVariableDim = 4; // cx, cy, dx, dy
	stateMeasureDim = 2;  // cx, cy
	nextID = 0;
	m_thres_invisibleCnt = 6;
	n_velocity_deque = 6;
	n_orientation_deque = 6;
	thr_velocity = 10; // m/s
	thr_orientation = M_PI / 6; // 30 degree

	//A & Q ==> Predict process
	//H & R ==> Estimation process
	
	// dt = 0.05; // 0.01

	// A
	m_matTransition = (Mat_<float>(stateVariableDim, stateVariableDim) << 1, 0, dt, 0,
																		  0, 1, 0, dt,
																		  0, 0, 1, 0,
																		  0, 0, 0, 1);

	// H
	m_matMeasurement = (Mat_<float>(stateMeasureDim, stateVariableDim) << 1, 0, 0, 0,
																		  0, 1, 0, 0);

	// Q size small -> smooth
	// float Q[] = {1e-5f, 1e-3f, 1e-3f, 1e-3f};
	float Q[] = {1e-2f, 1e-2f, 1e-2f, 1e-2f};
	Mat tempQ(stateVariableDim, 1, CV_32FC1, Q);
	m_matProcessNoiseCov = Mat::diag(tempQ);

	// R
	float R[] = {1e-3f, 1e-3f};
	Mat tempR(stateMeasureDim, 1, CV_32FC1, R);
	m_matMeasureNoiseCov = Mat::diag(tempR);

	m_thres_associationCost = 5.0f;
}

//deconstructor
Track::~Track(){}

void Track::velocity_push_back(std::deque<float> &deque, float v) 
{
	if (deque.size() < n_velocity_deque) { deque.push_back(v); }
	else { // 제로백 5초 기준 가속도는 5.556m/(s*)
		if (abs(deque.back() - v) < thr_velocity) { deque.push_back(v); }
		else { deque.push_back(0.0); }
		deque.pop_front(); }
}

void Track::orientation_push_back(std::deque<float> &deque, float o)
{
    // 현재 deque 크기가 max_size 미만이면 새로운 값을 추가
    if (deque.size() < n_orientation_deque) {
        deque.push_back(o);
    }
    else {
        // 이전 값들의 평균 계산
        float sum_yaw = 0.0;
        for (size_t i = 0; i < deque.size(); ++i) {
            sum_yaw += deque[i];
        }
        float avg_yaw = sum_yaw / deque.size();

        // 새로운 값과 평균의 차이 계산 (각도 차이의 절댓값)
        float yaw_diff = std::fabs(angles::shortest_angular_distance(avg_yaw, o));

        // 임계값 설정 (예: 30도)
        float yaw_threshold = thr_orientation; // 30도를 라디안으로 변환

        if (yaw_diff <= yaw_threshold) {
            // 차이가 임계값 이내이면 새로운 값을 추가
            deque.push_back(o);
        }
        else {
            // 차이가 너무 크면 이전 값을 다시 추가하여 deque 크기 유지
            deque.push_back(deque.back());
        }

		deque.pop_front();
    }
}



float Track::getVectorScale(float v1, float v2)
{
	float distance = sqrt(pow(v1, 2) + pow(v2, 2));
	if (v1 < 0) return -distance;
	else return distance;
}

double Track::getBBoxRatio(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
 	
 	double boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
	double boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

	double gap;
	if(boxAArea > boxBArea) gap = boxAArea / boxBArea;
	else gap = boxBArea / boxAArea;

	return gap;
}

double Track::getBBoxDistance(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{	
	float distance = sqrt(pow(bbox2.pose.position.x - bbox1.pose.position.x, 2) + pow(bbox2.pose.position.y - bbox1.pose.position.y, 2));
	return distance;
}

visualization_msgs::Marker Track::get_text_msg(struct trackingStruct &track, int i)
{
	visualization_msgs::Marker text;
	text.ns = "text";
	text.id = i;
	text.action = visualization_msgs::Marker::ADD;
	text.lifetime = ros::Duration(0.05);
	text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	text.color.r = 1.0;
	text.color.g = 1.0;
	text.color.b = 1.0;
	text.color.a = 1.0;
	text.scale.z = 1.0;

	text.pose.position.x = track.cur_bbox.pose.position.x;
	text.pose.position.y = track.cur_bbox.pose.position.y;
	text.pose.position.z = track.cur_bbox.pose.position.z + 2.0;
	text.pose.orientation.w = 1.0;

	char buf[100];
	//sprintf(buf, "ID : %d", track.id*10+1);
	//sprintf(buf, "%f", text.pose.position.y);
	sprintf(buf, "ID: %d\nAge: %d\nV: %dkm/h", track.id, track.age ,int(track.v*3.6));

	text.text = buf;

	return text;
}

void Track::predictNewLocationOfTracks(const ros::Time &currentTime)
{
	for (int i = 0; i < vecTracks.size(); i++)
	{
		dt = currentTime.toSec() - vecTracks[i].lastTime;

		vecTracks[i].kf.transitionMatrix.at<float>(0, 2) = dt;
        vecTracks[i].kf.transitionMatrix.at<float>(1, 3) = dt;

		// Predict current state
		vecTracks[i].kf.predict();

		vecTracks[i].cur_bbox.pose.position.x = vecTracks[i].kf.statePre.at<float>(0);
		vecTracks[i].cur_bbox.pose.position.y = vecTracks[i].kf.statePre.at<float>(1);

		vecTracks[i].vx = vecTracks[i].kf.statePre.at<float>(2);
		vecTracks[i].vy = vecTracks[i].kf.statePre.at<float>(3);
	}
}

void Track::assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{
	int N = (int)vecTracks.size();             //  N = number of tracking
	int M = (int)bboxArray.boxes.size(); //  M = number of detection

	vector<vector<double>> Cost(N, vector<double>(M)); //2 array

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			// Distance
			Cost[i][j] = getBBoxDistance(vecTracks[i].cur_bbox, bboxArray.boxes[j]);
		}
	}

	vector<int> assignment;

	if (N != 0)
	{
		AssignmentProblemSolver APS;
		APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);
	}

	vecAssignments.clear();
	vecUnassignedTracks.clear();
	vecUnssignedDetections.clear();

	for (int i = 0; i < N; i++)
	{
		if (assignment[i] == -1)
		{
			vecUnassignedTracks.push_back(i);
		}
		else
		{
			if ((Cost[i][assignment[i]] < m_thres_associationCost) && (3 > getBBoxRatio(vecTracks[i].cur_bbox, bboxArray.boxes[assignment[i]])))
			{
				vecAssignments.push_back(pair<int, int>(i, assignment[i]));
			}
			else
			{
				vecUnassignedTracks.push_back(i);
				assignment[i] = -1;
			}
		}
	}

	for (int j = 0; j < M; j++)
	{
		auto it = find(assignment.begin(), assignment.end(), j);
		if (it == assignment.end())
			vecUnssignedDetections.push_back(j);
	}
}

// 상대 좌표계
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{	
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;
		int idD = vecAssignments[i].second;

		Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
		measure.at<float>(0) = bboxArray.boxes[idD].pose.position.x;
		measure.at<float>(1) = bboxArray.boxes[idD].pose.position.y;
		vecTracks[idT].kf.correct(measure);
		
		velocity_push_back(vecTracks[idT].vx_deque, vecTracks[idT].kf.statePost.at<float>(2));
		velocity_push_back(vecTracks[idT].vy_deque, vecTracks[idT].kf.statePost.at<float>(3));
		// velocity_push_back(vecTracks[idT].v_deque, getVectorScale(vecTracks[idT].kf.statePost.at<float>(2), vecTracks[idT].kf.statePost.at<float>(3)));
		velocity_push_back(vecTracks[idT].v_deque, std::hypot(vecTracks[idT].kf.statePost.at<float>(2), vecTracks[idT].kf.statePost.at<float>(3)));
		vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0) / vecTracks[idT].vx_deque.size();
		vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0) / vecTracks[idT].vy_deque.size();
		vecTracks[idT].v = std::accumulate(vecTracks[idT].v_deque.begin(), vecTracks[idT].v_deque.end(), 0.0) / vecTracks[idT].v_deque.size();

		// 이전 orientation들과 비교
        orientation_push_back(vecTracks[idT].orientation_deque, tf::getYaw(bboxArray.boxes[idD].pose.orientation));
		vecTracks[idT].cur_bbox.pose.orientation = tf::createQuaternionMsgFromYaw(vecTracks[idT].orientation_deque.back());

		// 0.5초 이상 추적 된 객체가 딥러닝 기반일 시, 딥러닝 정보를 우선적으로 사용
		if (bboxArray.boxes[idD].label == 0 && vecTracks[idT].pre_bbox.label != 0 && vecTracks[idT].age > 10) {
			vecTracks[idT].cur_bbox.dimensions = vecTracks[idT].pre_bbox.dimensions;
			vecTracks[idT].cur_bbox.pose.orientation = vecTracks[idT].pre_bbox.pose.orientation; 
			vecTracks[idT].cur_bbox.label = vecTracks[idT].pre_bbox.label; }
		else {
			vecTracks[idT].cur_bbox.dimensions = bboxArray.boxes[idD].dimensions;
			vecTracks[idT].cur_bbox.pose.orientation = bboxArray.boxes[idD].pose.orientation;
			vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label; }
		
		vecTracks[idT].cur_bbox.pose.position = bboxArray.boxes[idD].pose.position;
		// vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label;
		vecTracks[idT].cur_bbox.header.stamp = bboxArray.boxes[idD].header.stamp; // 중요, 이거 안 하면 time stamp 안 맞아서 스탬프가 오래됐을 경우 rviz에서 제대로 표시 안됨

		vecTracks[idT].pre_bbox = vecTracks[idT].cur_bbox;
		// vecTracks[idT].sec = lidarSec.toSec();
		vecTracks[idT].lastTime = bboxArray.boxes[idD].header.stamp.toSec();
		vecTracks[idT].age++;
		vecTracks[idT].cntTotalVisible++;
		vecTracks[idT].cntConsecutiveInvisible = 0;
	}
}

void Track::unassignedTracksUpdate()
{
	for (int i = 0; i < (int)vecUnassignedTracks.size(); i++)
	{
		int id = vecUnassignedTracks[i];
		vecTracks[id].age++;
		vecTracks[id].cntConsecutiveInvisible++;
	}
}

void Track::deleteLostTracks()
{
	if ((int)vecTracks.size() == 0)
	{
		return;
	}

	for (int i = vecTracks.size() - 1; i >= 0; i--)
	{
		if (vecTracks[i].cntConsecutiveInvisible >= m_thres_invisibleCnt)
		{
			vecTracks.erase(vecTracks.begin() + i);
		}
	}

}

void Track::createNewTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{
	for (int i = 0; i < (int)vecUnssignedDetections.size(); i++)
	{
		int id = vecUnssignedDetections[i];

		trackingStruct ts;
		ts.id = nextID++;
		ts.age = 1;
		ts.cntTotalVisible = 1;
		ts.cntConsecutiveInvisible = 0;

		ts.cur_bbox = bboxArray.boxes[id];
		ts.pre_bbox = bboxArray.boxes[id];

		ts.vx = 0.0;
		ts.vy = 0.0;
		ts.v = 0.0;
		ts.sec = 0.0;

		ts.kf.init(stateVariableDim, stateMeasureDim);

		m_matTransition.copyTo(ts.kf.transitionMatrix);         //A
		m_matMeasurement.copyTo(ts.kf.measurementMatrix);       //H

		m_matProcessNoiseCov.copyTo(ts.kf.processNoiseCov);     //Q
		m_matMeasureNoiseCov.copyTo(ts.kf.measurementNoiseCov); //R

		Mat tempCov(stateVariableDim, 1, CV_32FC1, 1);
		ts.kf.errorCovPost = Mat::diag(tempCov);

		ts.kf.statePost.at<float>(0) = ts.cur_bbox.pose.position.x;
		ts.kf.statePost.at<float>(1) = ts.cur_bbox.pose.position.y;
		ts.kf.statePost.at<float>(2) = ts.vx;
		ts.kf.statePost.at<float>(3) = ts.vy;

		ts.lastTime = bboxArray.boxes[id].header.stamp.toSec();
		
		vecTracks.push_back(ts);
	}
}

pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> Track::displayTrack()
{   
	jsk_recognition_msgs::BoundingBoxArray bboxArray;
	visualization_msgs::MarkerArray textArray;
	
	for (int i = 0; i < vecTracks.size(); i++)
	{
		vecTracks[i].cur_bbox.header.seq = vecTracks[i].age; // header.seq를 tracking object의 age로 사용
		if (vecTracks[i].age >= 2 && vecTracks[i].cntConsecutiveInvisible == 0)
		{
			// float interpolation_time = ros::Time::now().toSec() - vecTracks[i].sec; // mingu
			// vecTracks[i].cur_bbox.pose.position.x += vecTracks[i].vx * interpolation_time;
			// vecTracks[i].cur_bbox.pose.position.y += vecTracks[i].vy * interpolation_time;

			vecTracks[i].cur_bbox.value = vecTracks[i].v;
			bboxArray.boxes.push_back(vecTracks[i].cur_bbox);
			textArray.markers.push_back(get_text_msg(vecTracks[i], i));
		}
	}
	
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_marker(bboxArray, textArray);
	return bbox_marker;
}