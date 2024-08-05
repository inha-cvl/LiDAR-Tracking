#include "track/track.h"
#include <cmath>

using namespace cv;

std::pair<float, float> getIQRThreshold(const std::vector<float> &data, float multiplier)
{
    std::vector<float> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());

    int dataSize = sortedData.size();
    int q1Index = dataSize / 4;
    int q3Index = 3 * dataSize / 4;

    float q1 = sortedData[q1Index];
    float q3 = sortedData[q3Index];

    float iqr = q3 - q1;
    float lowerThreshold = q1 - multiplier * iqr;
    float upperThreshold = q3 + multiplier * iqr;

    // return std::max(lowerThreshold, 0.0f); // Ensure the threshold is non-negative
	return std::make_pair(lowerThreshold, upperThreshold);
}

//constructor
Track::Track()
{
	stateVariableDim = 4; // cx, cy, dx, dy
	stateMeasureDim = 2;  // cx, cy
	nextID = 0;
	m_thres_invisibleCnt = 3;

	//A & Q ==> Predict process
	//H & R ==> Estimation process
	
	dt = 0.1;

	// A
	m_matTransition = (Mat_<float>(stateVariableDim, stateVariableDim) << 1, 0, dt, 0,
																		  0, 1, 0, dt,
																		  0, 0, 0, 0,
																		  0, 0, 0, 0);

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

void Track::deque_push_back(std::deque<float> &deque, float v) 
{
	if (deque.size() < 4) { deque.push_back(v); }
	else { // 제로백 5초 기준 가속도는 5.556m/(s*)
		if (abs(deque.back() - v) < 10) { deque.push_back(v); }
		else { deque.push_back(0.0); }
		deque.pop_front(); }
}

float Track::getVectorScale(float v1, float v2)
{
	float distance = sqrt(pow(v1, 2) + pow(v2, 2));
	if (v1 < 0) return -distance;
	else return distance;
}

double Track::getBBoxIOU(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
	double xA = max(boxA[0], boxB[0]);
	double yA = max(boxA[1], boxB[1]);
	double xB = min(boxA[2], boxB[2]);
	double yB = min(boxA[3], boxB[3]);

	double interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1);
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	double boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	double iou = interArea / double(boxAArea + boxBArea - interArea);

	return iou;
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
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
	double boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

	double gap;
	if(boxAArea > boxBArea) gap = boxAArea / boxBArea;
	else gap = boxBArea / boxAArea;

	return gap;
}

double Track::getBBoxArea(jsk_recognition_msgs::BoundingBox bbox1)
{
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);

	return boxAArea;
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
	text.lifetime = ros::Duration(0.2);
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

void Track::predictNewLocationOfTracks()
{
	for (int i = 0; i < vecTracks.size(); i++)
	{
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
			// Box Over Lap
			// Cost[i][j] = 1 - getBBoxIOU(vecTracks[i].cur_bbox, bboxArray.boxes[j]);
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
	vecUnssignedTracks.clear();
	vecUnssignedDetections.clear();

	for (int i = 0; i < N; i++)
	{
		if (assignment[i] == -1)
		{
			vecUnssignedTracks.push_back(i);
		}
		else
		{
			if ((Cost[i][assignment[i]] < m_thres_associationCost) && (3 > getBBoxRatio(vecTracks[i].cur_bbox, bboxArray.boxes[assignment[i]])))
			{
				vecAssignments.push_back(pair<int, int>(i, assignment[i]));
			}
			else
			{
				vecUnssignedTracks.push_back(i);
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

std::tuple<double, double> Track::convertAbsoluteCoordinates(double absoluteX, double absoluteY, double angle, double trackX, double trackY)
{
	double x, y, theta;
	theta = angle * (M_PI / 180.0);
	x = absoluteX + (trackX * cos(theta) - trackY * sin(theta));
	y = absoluteY + (trackX * sin(theta) + trackY * cos(theta));

	return std::make_tuple(x,y);
}

// 상대 좌표계
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray, const ros::Time &lidarSec)
{	
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;
		int idD = vecAssignments[i].second;

		Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
		measure.at<float>(0) = bboxArray.boxes[idD].pose.position.x;
		measure.at<float>(1) = bboxArray.boxes[idD].pose.position.y;
		vecTracks[idT].kf.correct(measure);
		
		float vx = (bboxArray.boxes[idD].pose.position.x - vecTracks[idT].pre_bbox.pose.position.x) / dt;
		float vy = (bboxArray.boxes[idD].pose.position.y - vecTracks[idT].pre_bbox.pose.position.y) / dt;
		float v = getVectorScale(vx, vy);

		deque_push_back(vecTracks[idT].vx_deque, vx);
		deque_push_back(vecTracks[idT].vy_deque, vy);
		deque_push_back(vecTracks[idT].v_deque, v);

		vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0) / vecTracks[idT].vx_deque.size();
		vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0) / vecTracks[idT].vy_deque.size();
		vecTracks[idT].v = std::accumulate(vecTracks[idT].v_deque.begin(), vecTracks[idT].v_deque.end(), 0.0) / vecTracks[idT].v_deque.size();

		//vecTracks[idT].cur_bbox.header.stamp = ros::Time::now(); // mingu 중요, 이거 안 하면 time stamp 안 맞아서 스탬프가 오래됐을 경우 rviz에서 제대로 표시 안됨
		vecTracks[idT].cur_bbox.header.stamp = bboxArray.boxes[idD].header.stamp; // 쩔수없이..

		//std::cout << bboxArray.boxes[idD].header.stamp << std::endl;

		if (bboxArray.boxes[idD].label == 0 && vecTracks[idT].pre_bbox.label != 0) {
			vecTracks[idT].cur_bbox.dimensions = vecTracks[idT].pre_bbox.dimensions;
			vecTracks[idT].cur_bbox.pose.orientation = vecTracks[idT].pre_bbox.pose.orientation; }
		else {
			vecTracks[idT].cur_bbox.dimensions = bboxArray.boxes[idD].dimensions;
			vecTracks[idT].cur_bbox.pose.orientation = bboxArray.boxes[idD].pose.orientation; }
		
		vecTracks[idT].cur_bbox.pose.position = bboxArray.boxes[idD].pose.position;

		vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label; // mingu
		
		vecTracks[idT].pre_bbox = vecTracks[idT].cur_bbox;
		vecTracks[idT].sec = lidarSec.toSec();
		vecTracks[idT].age++;
		vecTracks[idT].cntTotalVisible++;
		vecTracks[idT].cntConsecutiveInvisible = 0;
	}
}

void Track::unassignedTracksUpdate()
{
	for (int i = 0; i < (int)vecUnssignedTracks.size(); i++)
	{
		int id = vecUnssignedTracks[i];
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
	for (int i = 0; i < (int)vecTracks.size(); i++)
	{
		if (vecTracks[i].cntConsecutiveInvisible >= m_thres_invisibleCnt)
		{
			vecTracks.erase(vecTracks.begin() + i);
			i--;
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