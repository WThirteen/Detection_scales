#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;

#define PI 3.1415926
Mat Corner_detection(Mat img) {
	float row = img.rows;
	float col = img.cols;
	float c = img.channels();
	Mat img1;
	float k = 3;
	resize(img, img1, Size(round(row / k), round(col / k)));
	//cout << "row=" << round(row / k) << "  " << "col=" << round(col / k) << endl;
	Mat gray, gray1;
	if (c > 1) {
		cvtColor(img, gray, COLOR_BGR2GRAY);
	}

	gray.convertTo(gray, CV_32FC1);
	resize(gray, gray1, Size(round(row / k), round(col / k)));
	//�����С��һ��

	float min_gray1 = *(min_element(gray1.begin<float>(), gray1.end<float>()));
	float max_gray1 = *(max_element(gray1.begin<float>(), gray1.end<float>()));
	//cout << "min_gray=" << min_gray << "  " << "max_gray=" << max_gray << endl;

	gray1 = (gray1 - min_gray1) / (max_gray1 - min_gray1 + 0.001);
	//�����˲�ģ��
	Mat mask_x = (Mat_<float>(1, 3) << -1, 0, 1);
	Mat mask_y = (Mat_<float>(3, 1) << -1, 0, 1);
	Mat X, Y, X2, Y2, XY;
	//�ֱ����x��y�����ƫ����--�˲�
	filter2D(gray1, X, CV_32FC1, mask_x);
	filter2D(gray1, Y, CV_32FC1, mask_y);
	//����Э�������M:M=[A C; C B]
	X2 = X.mul(X);
	Y2 = Y.mul(Y);
	XY = X.mul(Y);

	Mat gauss = (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16;
	//cout << "gauss=" << gauss << endl;
	Mat A, B, C;
	filter2D(X2, A, CV_32FC1, gauss);
	filter2D(Y2, B, CV_32FC1, gauss);
	filter2D(XY, C, CV_32FC1, gauss);

	//����ǵ���ӦR
	float row1 = gray1.rows;
	float col1 = gray1.cols;

	Mat det_M, trace_M, R;
	det_M = A.mul(B) - C.mul(C);
	trace_M = A + B;
	R = det_M - 0.04 * trace_M.mul(trace_M);
	Mat R1 = trace_M.mul(trace_M) / (det_M + 0.001); //��֤������Ϊ0
	//�ҽǵ�
	float max_R = *(max_element(R.begin<float>(), R.end<float>()));
	//cout << "max_R=" << max_R << endl;
	Mat temp;
	float max_temp;
	Point point;//opencv�ṩ�� �ýṹ���ʾ������
	//�ҽǵ�
	vector<float> pix; //�м����
	vector<vector<float>> pix_corner; //��¼�ǵ�����
	for (int i = 0; i < row1 - 2; i++) {
		for (int j = 0; j < col1 - 2; j++) {
			temp = R(Rect(j, i, 3, 3));//�������ڴ�С
			max_temp = *(max_element(temp.begin<float>(), temp.end<float>()));
			if (max_temp >= 0.03 * max_R && R.at<float>(i, j) == max_temp && R1.at<float>(i, j) > 2 && R1.at<float>(i, j) < 7) { //������ֵ(k*max_R)��Ϊ�ǽǵ� 
				point.y = i; //������point��ֵ
				point.x = j;
				pix.push_back(i);
				pix.push_back(j);
				pix_corner.push_back(pix);//��¼������
				pix.clear(); //��֤ÿһ��ѭ����pix�������ǿ�ֵ
				circle(img1, point, 1, Scalar(0, 255, 255));
			}
		}
	}
	for (vector<vector<float>>::iterator it = pix_corner.begin(); it != pix_corner.end(); it++) {
		vector<float>::iterator iti = (*it).begin();//ÿ�е��׵�ַ
		float x = *iti; // ������AAA
		iti++;
		float y = *iti; //������
		if (y - 10 > 0 && x - 10 > 0 && y + 10 < col1 && x + 10 < row1) { //����߽�����
			Mat temp_window = gray1(Rect(y - 10, x - 10, 21, 21)); //
		}
	}
	return img1;
}
void draw_line(Mat img, float rou, float theta, float row1) {
	Point p1, p2;
	rou = rou - row1;
	p1.x = round(rou * cos(theta) + row1 * (-sin(theta)));
	p1.y = round(rou * sin(theta) + row1 * (cos(theta)));
	p2.x = round(rou * cos(theta) - row1 * (-sin(theta)));
	p2.y = round(rou * sin(theta) - row1 * (cos(theta)));
	Scalar color(0, 255, 0);
	line(img, p1, p2, color, 2);
}

float find_max(Mat mat) {
	double temp = -1;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			if (temp < mat.at<float>(i, j)) {
				temp = mat.at<float>(i, j);
			}
		}
	}
	return temp;
}

Mat Hough_straight_line(Mat img1) {
	float row = img1.rows;
	float col = img1.cols;
	float r = img1.channels();
	Mat gray;
	if (r > 1) {
		cvtColor(img1, gray, COLOR_BGR2GRAY);
	}
	gray.convertTo(gray, CV_32FC1);
	//ʹ��canny ��ֵ��
	Mat canny_img;
	//canny�����ͼ��ΪCV_8UC1��ת��ΪCV_32FC1
	Canny(img1, canny_img, 60, 120, 3);
	canny_img.convertTo(canny_img, CV_32FC1);
	//3Ϊsobel�˲�ģ���С
	float row1 = round(sqrt(row * row + col * col));
	float col1 = 180;
	Mat hough_mat = Mat::zeros(row1 * 2, col1, CV_32FC1);

	int temp;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (canny_img.at<float>(i, j) != 0) {
				for (int theta = 0; theta < 180; theta++) {
					//temp = i * cos(theta * PI / 180) + j * sin(theta * PI / 180);
					temp = i * sin(theta * PI / 180) + j * cos(theta * PI / 180);
					// ͶƱ
					hough_mat.at<float>(round(temp) + row1, theta)++;
				}
			}
		}
	}
	//�����洢ֱ�߲����Ķ�ά����
	vector<Vec2f> lines;
	//�ȼ���vector<vector<float>> lines;
	vector<Vec2f> lines2;
	//myHoughLines(img1, 1, 1, 100, lines2);
	double max_judge = find_max(hough_mat);
	int judge1 = max_judge * 0.19;
	int judge2 = max_judge * 0.195;
	//����hongh����
	for (int i = 0; i < 2 * row1 - 2; i++) {
		for (int j = 0; j < 180 - 2; j++) {
			if (hough_mat.at<float>(i, j) > judge1 && hough_mat.at<float>(i, j) < judge2) {
				//���������ֵ��rou
				Vec2f point(i, j * PI / 180);
				//�˲�����3*3
				hough_mat.at<float>(i, j + 1) = 0;
				hough_mat.at<float>(i + 1, j) = 0;
				hough_mat.at<float>(i + 1, j + 1) = 0;
				hough_mat.at<float>(i + 1, j + 2) = 0;
				hough_mat.at<float>(i + 2, j) = 0;
				hough_mat.at<float>(i + 2, j + 1) = 0;
				hough_mat.at<float>(i + 2, j + 2) = 0;
				lines.push_back(point);
				//��i��j�Ļ����ƴ��
			}
		}
	}
	//cout << "line1(�˲������)" << endl;

	//ʹ��line1����ͼ��
	for (int i = 0; i < size(lines); i++) {
		draw_line(img1, lines[i][0], lines[i][1], row1);
	}
	return img1;
}



int main() {
	Mat img = imread("D:/�����ļ���/ѧϰ����/ͼ����������/week_1/�Ͽβ���/6.jpg");
	Mat img1 = Corner_detection(img);
	imshow("�ǵ���", img1);
	Mat img2 = Hough_straight_line(img1);
	imshow("�ǵ���+hough", img2);
	waitKey(0);
	return 0;
}