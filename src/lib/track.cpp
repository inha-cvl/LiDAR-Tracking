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
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray)
{	
	for (int i = 0; i < (int)vecAssignments.size(); i++)
	{
		int idT = vecAssignments[i].first;
		int idD = vecAssignments[i].second;

        // 위치 변화로부터 속도 계산
		float dx = bboxArray.boxes[idD].pose.position.x - vecTracks[idT].pre_bbox.pose.position.x;
		float dy = bboxArray.boxes[idD].pose.position.y - vecTracks[idT].pre_bbox.pose.position.y;

        float vx = dx / dt;
        float vy = dy / dt;
        
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
		
		// vecTracks[idT].v = getVectorScale(vecTracks[idT].kf.statePost.at<float>(2), vecTracks[idT].kf.statePost.at<float>(3));
		vecTracks[idT].v = getVectorScale(vecTracks[idT].vx, vecTracks[idT].vy);
		// vecTracks[idT].v = vecTracks[idT].kf.statePost.at<float>(2);
		// vecTracks[idT].v = vecTracks[idT].vx;
		
		float k = 300.0f;  // 비례 상수, 필요에 따라 조정 가능

		// dx의 제곱을 이용한 가변적인 scaling factor 계산
		// float scaling_factor = std::min(1 + k * std::pow(std::abs(dx), 2), 6.0);
		float scaling_factor = std::min(1 + k * std::pow(std::abs(sqrt(pow(dx, 2) + pow(dy, 2))), 2), 6.0);

		vecTracks[idT].v = vecTracks[idT].v * scaling_factor;

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

// 전역 좌표계
/*
geometry_msgs::PoseStamped previous_enu_pose;
bool has_previous_enu_pose = false;  // 초기 상태를 나타내기 위한 플래그
void Track::assignedTracksUpdate(const jsk_recognition_msgs::BoundingBoxArray &bboxArray, const geometry_msgs::PoseStamped &enu_pose)
{
    // 현재 ENU 좌표와 Yaw(헤딩 방향)를 가져옵니다.
    float enu_x_curr = enu_pose.pose.position.x;
    float enu_y_curr = enu_pose.pose.position.y;

    // 현재 Yaw를 디그리에서 라디안으로 변환
    float yaw_curr = enu_pose.pose.orientation.z * M_PI / 180.0f;

    // 이전 ENU 좌표와 Yaw를 가져옵니다.
    float enu_x_prev, enu_y_prev, yaw_prev;
    if (has_previous_enu_pose)
    {
        enu_x_prev = previous_enu_pose.pose.position.x;
        enu_y_prev = previous_enu_pose.pose.position.y;
        yaw_prev = previous_enu_pose.pose.orientation.z * M_PI / 180.0f;  // 디그리 -> 라디안 변환
    }
    else
    {
        // 이전 ENU 포즈가 없는 경우, 현재 값을 사용합니다.
        enu_x_prev = enu_x_curr;
        enu_y_prev = enu_y_curr;
        yaw_prev = yaw_curr;
    }

    for (int i = 0; i < static_cast<int>(vecAssignments.size()); i++)
    {
        int idT = vecAssignments[i].first;  // 트랙 ID
        int idD = vecAssignments[i].second; // 감지된 객체 ID

        // 현재 감지된 객체의 상대 좌표
        float relative_x_curr = bboxArray.boxes[idD].pose.position.x;
        float relative_y_curr = bboxArray.boxes[idD].pose.position.y;

        // 이전 감지된 객체의 상대 좌표
        float relative_x_prev = vecTracks[idT].pre_bbox.pose.position.x;
        float relative_y_prev = vecTracks[idT].pre_bbox.pose.position.y;

        // yaw를 고려하여 현재 감지된 객체의 전역 좌표로 변환
        float global_x_curr = enu_x_curr + (cos(yaw_curr) * relative_x_curr - sin(yaw_curr) * relative_y_curr);
        float global_y_curr = enu_y_curr + (sin(yaw_curr) * relative_x_curr + cos(yaw_curr) * relative_y_curr);

        // yaw를 고려하여 이전 감지된 객체의 전역 좌표로 변환
        float global_x_prev = enu_x_prev + (cos(yaw_prev) * relative_x_prev - sin(yaw_prev) * relative_y_prev);
        float global_y_prev = enu_y_prev + (sin(yaw_prev) * relative_x_prev + cos(yaw_prev) * relative_y_prev);

        // 이전 위치로부터 속도 계산 (전역 좌표계 기준)
        float vx = (global_x_curr - global_x_prev) / dt;
        float vy = (global_y_curr - global_y_prev) / dt;

        // 속도 deque 업데이트
        velocity_push_back(vecTracks[idT].vx_deque, vx);
        velocity_push_back(vecTracks[idT].vy_deque, vy);
        vecTracks[idT].vx = std::accumulate(vecTracks[idT].vx_deque.begin(), vecTracks[idT].vx_deque.end(), 0.0f) / vecTracks[idT].vx_deque.size();
        vecTracks[idT].vy = std::accumulate(vecTracks[idT].vy_deque.begin(), vecTracks[idT].vy_deque.end(), 0.0f) / vecTracks[idT].vy_deque.size();

        // 측정값(상대 좌표와 속도)을 Kalman 필터에 적용
        Mat measure = Mat::zeros(stateMeasureDim, 1, CV_32FC1);
        measure.at<float>(0) = relative_x_curr;    // 상대 x 좌표
        measure.at<float>(1) = relative_y_curr;    // 상대 y 좌표
        measure.at<float>(2) = vecTracks[idT].vx;  // x 방향 속도 (전역 좌표계)
        measure.at<float>(3) = vecTracks[idT].vy;  // y 방향 속도 (전역 좌표계)

        // Kalman 필터 보정
        vecTracks[idT].kf.correct(measure);

        // 속도의 크기 계산 (전역 좌표계 속도 기반)
        vecTracks[idT].v = sqrt(pow(vecTracks[idT].kf.statePost.at<float>(2), 2) + pow(vecTracks[idT].kf.statePost.at<float>(3), 2));

        // 위치 정보는 상대 좌표계를 그대로 유지
        vecTracks[idT].cur_bbox.pose.position = bboxArray.boxes[idD].pose.position;

        // Orientation 업데이트 (상대 좌표계 기준)
        orientation_push_back(vecTracks[idT].orientation_deque, tf::getYaw(bboxArray.boxes[idD].pose.orientation));
        vecTracks[idT].cur_bbox.pose.orientation = tf::createQuaternionMsgFromYaw(vecTracks[idT].orientation_deque.back());

        // 딥러닝 기반 정보 우선 사용 로직은 기존과 동일하게 유지
        if (bboxArray.boxes[idD].label == 0 && vecTracks[idT].pre_bbox.label != 0 && vecTracks[idT].age > m_thres_invisibleCnt) {
            vecTracks[idT].cur_bbox.dimensions = vecTracks[idT].pre_bbox.dimensions;
            vecTracks[idT].cur_bbox.pose.orientation = vecTracks[idT].pre_bbox.pose.orientation;
            vecTracks[idT].cur_bbox.label = vecTracks[idT].pre_bbox.label;
        } else {
            vecTracks[idT].cur_bbox.dimensions = bboxArray.boxes[idD].dimensions;
            vecTracks[idT].cur_bbox.pose.orientation = bboxArray.boxes[idD].pose.orientation;
            vecTracks[idT].cur_bbox.label = bboxArray.boxes[idD].label;
        }

        vecTracks[idT].cur_bbox.header.stamp = bboxArray.boxes[idD].header.stamp;

        // 이전 상태 업데이트
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

    // 현재의 enu_pose를 저장하여 다음 프레임에서 이전 enu_pose로 사용
    previous_enu_pose = enu_pose;
    has_previous_enu_pose = true;
}
*/

void Track::unassignedTracksUpdate()
{
	for (int i = 0; i < (int)vecUnassignedTracks.size(); i++)
	{
		int id = vecUnassignedTracks[i];
		vecTracks[id].age++;
		vecTracks[id].cntConsecutiveInvisible++;

		// 객체 유지
		vecTracks[id].cur_bbox = vecTracks[id].pre_bbox;
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