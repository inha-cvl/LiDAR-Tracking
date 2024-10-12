#include "track/track.h"
#include <cmath>

using namespace cv;

//constructor
Track::Track()
{
	stateVariableDim = 6; // cx, cy, dx, dy
	stateMeasureDim = 4;
	nextID = 0;
	m_thres_invisibleCnt = 6;
	n_velocity_deque = 6;
	n_orientation_deque = 6;
	thr_velocity = 15; // m/s
	thr_orientation = M_PI / 6; // 30 degree

	//A & Q ==> Predict process
	//H & R ==> Estimation process
	
	// A 상태
	m_matTransition = Mat::eye(stateVariableDim, stateVariableDim, CV_32F);
	m_matTransition.at<float>(0, 2) = dt;          // x 위치는 vx에 비례
	m_matTransition.at<float>(0, 4) = 0.5 * dt * dt; // x 위치는 ax에 비례 (등가속도 반영)
	m_matTransition.at<float>(1, 3) = dt;          // y 위치는 vy에 비례
	m_matTransition.at<float>(1, 5) = 0.5 * dt * dt; // y 위치는 ay에 비례 (등가속도 반영)
	// 속도는 가속도에 의해 변화
	m_matTransition.at<float>(2, 4) = dt;          // vx는 ax에 비례
	m_matTransition.at<float>(3, 5) = dt;

	// H 관측
	m_matMeasurement = Mat::zeros(stateMeasureDim, stateVariableDim, CV_32F);
	m_matMeasurement.at<float>(0, 0) = 1.0f; // x
	m_matMeasurement.at<float>(1, 1) = 1.0f; // y
	m_matMeasurement.at<float>(2, 2) = 1.0f; // vx
	m_matMeasurement.at<float>(3, 3) = 1.0f; // vy

	


	// Q size small -> smooth 프로세스 노이즈
	float Q[] = {1e-3f, 1e-3f, 1e-2f, 1e-2f, 1e-1f, 1e-1f};
	Mat tempQ(stateVariableDim, 1, CV_32FC1, Q);
	m_matProcessNoiseCov = Mat::diag(tempQ);

	// R 관측 노이즈
	float R[] = {1e-3f, 1e-3f, 1e-2f, 1e-2f};
	Mat tempR(stateMeasureDim, 1, CV_32FC1, R);
	m_matMeasureNoiseCov = Mat::diag(tempR);

	m_thres_associationCost = 3.0f;
}

//deconstructor
Track::~Track(){}

void Track::velocity_push_back(std::deque<float> &deque, float v) 
{
	if (deque.size() < n_velocity_deque) { deque.push_back(v); }
	else { // 제로백 5초 기준 가속도는 5.556m/(s*)
		float sum_vel = 0.0;
        for (size_t i = 0; i < deque.size(); ++i) { sum_vel += deque[i]; }
        float avg_vel = sum_vel / deque.size();
		
		if (abs(avg_vel - v) < thr_velocity) { deque.push_back(v); }
		else { deque.push_back(deque.back()); }
		deque.pop_front(); }
}

void Track::orientation_push_back(std::deque<float> &deque, float o)
{
    if (deque.size() < n_orientation_deque) { deque.push_back(o); }
    else {
        float sum_yaw = 0.0;
        for (size_t i = 0; i < deque.size(); ++i) { sum_yaw += deque[i]; }

        float avg_yaw = sum_yaw / deque.size();
        float yaw_diff = std::fabs(angles::shortest_angular_distance(avg_yaw, o));

        if (yaw_diff <= thr_orientation) { deque.push_back(o); }
        else { deque.push_back(deque.back()); }

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
	// sprintf(buf, "ID: %d\nAge: %d\nV: %dkm/h", track.id, track.age ,int(track.v*3.6));
	// sprintf(buf, "Age: %d\nD: %dm\nV: %dkm/h", track.age ,int(text.pose.position.x), int(track.v*3.6));
	sprintf(buf, "Score: %.2f\nD: %dm\nV: %dkm/h", track.score ,int(text.pose.position.x), int(track.v*3.6));

	text.text = buf;

	return text;
}

void Track::predictNewLocationOfTracks(const ros::Time &currentTime)
{
    for (int i = 0; i < vecTracks.size(); i++)
    {
        dt = currentTime.toSec() - vecTracks[i].lastTime;

        // 상태 전이 행렬 업데이트 (등가속도 모델 반영)
        vecTracks[i].kf.transitionMatrix = Mat::eye(stateVariableDim, stateVariableDim, CV_32F);
        vecTracks[i].kf.transitionMatrix.at<float>(0, 2) = dt;
        vecTracks[i].kf.transitionMatrix.at<float>(0, 4) = 0.5f * dt * dt;
        vecTracks[i].kf.transitionMatrix.at<float>(1, 3) = dt;
        vecTracks[i].kf.transitionMatrix.at<float>(1, 5) = 0.5f * dt * dt;
        vecTracks[i].kf.transitionMatrix.at<float>(2, 4) = dt;
        vecTracks[i].kf.transitionMatrix.at<float>(3, 5) = dt;

        // 상태 예측
        vecTracks[i].kf.predict();

        // 예측된 위치와 속도 업데이트
        vecTracks[i].cur_bbox.pose.position.x = vecTracks[i].kf.statePre.at<float>(0);
        vecTracks[i].cur_bbox.pose.position.y = vecTracks[i].kf.statePre.at<float>(1);
        vecTracks[i].vx = vecTracks[i].kf.statePre.at<float>(2);
        vecTracks[i].vy = vecTracks[i].kf.statePre.at<float>(3);
        vecTracks[i].ax = vecTracks[i].kf.statePre.at<float>(4);
        vecTracks[i].ay = vecTracks[i].kf.statePre.at<float>(5);
    }
}

void Track::assignDetectionsTracks(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{
	int N = (int)vecTracks.size();       //  N = number of tracking
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
/*
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{	
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;
		int idD = vecAssignments[i].second;

        // 위치 변화로부터 속도 계산
        float vx = (bboxArray.boxes[idD].pose.position.x - vecTracks[idT].pre_bbox.pose.position.x) / dt;
        float vy = (bboxArray.boxes[idD].pose.position.y - vecTracks[idT].pre_bbox.pose.position.y) / dt;

		// 속도 deque 업데이트 및 평균 계산
        velocity_push_back(vecTracks[idT].vx_deque, vx);
        velocity_push_back(vecTracks[idT].vy_deque, vy);
        vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0f) / vecTracks[idT].vx_deque.size();
        vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0f) / vecTracks[idT].vy_deque.size();
		
		// 측정값 구성 (위치와 속도 포함)
        Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
        measure.at<float>(0) = bboxArray.boxes[idD].pose.position.x;
        measure.at<float>(1) = bboxArray.boxes[idD].pose.position.y;
        measure.at<float>(2) = vecTracks[idT].vx;
        measure.at<float>(3) = vecTracks[idT].vy;

        // 칼만 필터 보정
        vecTracks[idT].kf.correct(measure);

		vecTracks[idT].v = getVectorScale(vecTracks[idT].kf.statePost.at<float>(2), vecTracks[idT].kf.statePost.at<float>(3)); // 1
		// vecTracks[idT].v = getVectorScale(vecTracks[idT].vx, vecTracks[idT].vy); // 2
		// vecTracks[idT].v = vecTracks[idT].kf.statePost.at<float>(2);
		// vecTracks[idT].v = vecTracks[idT].vx; // 3
		
		// 이전 orientation들과 비교
        orientation_push_back(vecTracks[idT].orientation_deque, tf::getYaw(bboxArray.boxes[idD].pose.orientation));
		vecTracks[idT].cur_bbox.pose.orientation = tf::createQuaternionMsgFromYaw(vecTracks[idT].orientation_deque.back());

		// 0.5초 이상 추적 된 객체가 딥러닝 기반일 시, 딥러닝 정보를 우선적으로 사용
		if (bboxArray.boxes[idD].label == 0 && vecTracks[idT].pre_bbox.label != 0 && vecTracks[idT].age > m_thres_invisibleCnt) {
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
		vecTracks[idT].cls = vecTracks[idT].cur_bbox.label;
		if (vecTracks[idT].cls != 0) { vecTracks[idT].score = bboxArray.boxes[idD].value; }
		vecTracks[idT].lastTime = bboxArray.boxes[idD].header.stamp.toSec();
		vecTracks[idT].age++;
		vecTracks[idT].cntTotalVisible++;
		vecTracks[idT].cntConsecutiveInvisible = 0;
	}
}
*/
// 전역 좌표계
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray, const geometry_msgs::PoseStamped &enu_pose)
{
    // ENU 좌표와 Yaw(헤딩 방향)를 바탕으로 객체 추적을 업데이트
    float enu_x = enu_pose.pose.position.x;
    float enu_y = enu_pose.pose.position.y;

    // Yaw는 orientation.z로 주어지고, 디그리 단위이므로 라디안으로 변환
    float yaw = enu_pose.pose.orientation.z * M_PI / 180.0;  // 디그리 -> 라디안 변환

    for (int i = 0; i < static_cast<int>(vecAssignments.size()); i++)
    {
        int idT = vecAssignments[i].first;  // 트랙 ID
        int idD = vecAssignments[i].second; // 감지된 객체 ID

        // 상대 좌표 -> 전역 좌표로 변환 (ENU 기준)
        float relative_x = bboxArray.boxes[idD].pose.position.x;
        float relative_y = bboxArray.boxes[idD].pose.position.y;

        // yaw(헤딩 방향)를 고려하여 전역 좌표로 변환
        float global_x = enu_x + (cos(yaw) * relative_x - sin(yaw) * relative_y);
        float global_y = enu_y + (sin(yaw) * relative_x + cos(yaw) * relative_y);

        // 이전 위치로부터 속도 계산
        float prev_x = vecTracks[idT].cur_bbox.pose.position.x;
        float prev_y = vecTracks[idT].cur_bbox.pose.position.y;
        float vx = (global_x - prev_x) / dt;
        float vy = (global_y - prev_y) / dt;

        // 속도 deque 업데이트
        velocity_push_back(vecTracks[idT].vx_deque, vx);
        velocity_push_back(vecTracks[idT].vy_deque, vy);
        vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0f) / vecTracks[idT].vx_deque.size();
        vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0f) / vecTracks[idT].vy_deque.size();

        // 측정값(전역 좌표와 속도)을 Kalman 필터에 적용
        Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
        measure.at<float>(0) = global_x;  // 전역 x 좌표
        measure.at<float>(1) = global_y;  // 전역 y 좌표
        measure.at<float>(2) = vecTracks[idT].vx; // x 방향 속도
        measure.at<float>(3) = vecTracks[idT].vy; // y 방향 속도

        // Kalman 필터 보정
        vecTracks[idT].kf.correct(measure);

        // 추적된 객체의 새로운 위치 및 방향 업데이트
        vecTracks[idT].cur_bbox.pose.position.x = global_x;
        vecTracks[idT].cur_bbox.pose.position.y = global_y;
        orientation_push_back(vecTracks[idT].orientation_deque, yaw);
        vecTracks[idT].cur_bbox.pose.orientation = tf::createQuaternionMsgFromYaw(vecTracks[idT].orientation_deque.back());

        // 바운딩 박스 크기 및 기타 정보 업데이트
        vecTracks[idT].cur_bbox.dimensions = bboxArray.boxes[idD].dimensions;
        vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label;
        vecTracks[idT].cur_bbox.header.stamp = bboxArray.boxes[idD].header.stamp;

        // 이전 바운딩 박스 정보를 저장
        vecTracks[idT].pre_bbox = vecTracks[idT].cur_bbox;
        vecTracks[idT].cls = vecTracks[idT].cur_bbox.label;
        if (vecTracks[idT].cls != 0) {
            vecTracks[idT].score = bboxArray.boxes[idD].value;
        }
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

		// 객체 유지
		// vecTracks[id].cur_bbox = vecTracks[id].pre_bbox;
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
		ts.ax = 0.0;
		ts.ay = 0.0;


		ts.kf.init(stateVariableDim, stateMeasureDim);

		m_matTransition.copyTo(ts.kf.transitionMatrix);         //A
		m_matMeasurement.copyTo(ts.kf.measurementMatrix);       //H
		m_matProcessNoiseCov.copyTo(ts.kf.processNoiseCov);     //Q
		m_matMeasureNoiseCov.copyTo(ts.kf.measurementNoiseCov); //R

		// Mat tempCov(stateVariableDim, 1, CV_32FC1, 1);
		// tempCov.at<float>(0) = 1e-3f; // x 위치 불확실성
		// tempCov.at<float>(1) = 1e-3f; // y 위치 불확실성
		// tempCov.at<float>(2) = 1e-2f;  // vx 속도 불확실성
		// tempCov.at<float>(3) = 1e-2f;  // vy 속도 불확실성

		float P[] = {1e-3f, 1e-3f, 1e-2f, 1e-2f, 1e-1f, 1e-1f};
		Mat tempCov(stateVariableDim, 1, CV_32FC1, P);
		ts.kf.errorCovPost = Mat::diag(tempCov);

		ts.kf.statePost.at<float>(0) = ts.cur_bbox.pose.position.x;
		ts.kf.statePost.at<float>(1) = ts.cur_bbox.pose.position.y;
		ts.kf.statePost.at<float>(2) = ts.vx;
		ts.kf.statePost.at<float>(3) = ts.vy;
		ts.kf.statePost.at<float>(4) = 0.0f; // 초기 가속도는 0으로 설정
		ts.kf.statePost.at<float>(5) = 0.0f; // 초기 가속도는 0으로 설정

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
		// if ((vecTracks[i].age >= 2 && vecTracks[i].cntConsecutiveInvisible == 0) ||
		// 	(vecTracks[i].cls == 1 && vecTracks[i].age >= m_thres_invisibleCnt && vecTracks[i].cntConsecutiveInvisible <= m_thres_invisibleCnt/2))

		if (vecTracks[i].age >= 2 && vecTracks[i].cntConsecutiveInvisible == 0)
		{	
			vecTracks[i].cur_bbox.header.seq = vecTracks[i].age; // header.seq를 tracking object의 age로 사용
			vecTracks[i].cur_bbox.value = vecTracks[i].v;
			bboxArray.boxes.push_back(vecTracks[i].cur_bbox);
			textArray.markers.push_back(get_text_msg(vecTracks[i], i));
		}
	}
	
	pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox_marker(bboxArray, textArray);
	return bbox_marker;
}