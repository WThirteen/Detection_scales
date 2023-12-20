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
	//最大最小归一化

	float min_gray1 = *(min_element(gray1.begin<float>(), gray1.end<float>()));
	float max_gray1 = *(max_element(gray1.begin<float>(), gray1.end<float>()));
	//cout << "min_gray=" << min_gray << "  " << "max_gray=" << max_gray << endl;

	gray1 = (gray1 - min_gray1) / (max_gray1 - min_gray1 + 0.001);
	//创建滤波模板
	Mat mask_x = (Mat_<float>(1, 3) << -1, 0, 1);
	Mat mask_y = (Mat_<float>(3, 1) << -1, 0, 1);
	Mat X, Y, X2, Y2, XY;
	//分别计算x和y方向的偏导数--滤波
	filter2D(gray1, X, CV_32FC1, mask_x);
	filter2D(gray1, Y, CV_32FC1, mask_y);
	//计算协方差矩阵M:M=[A C; C B]
	X2 = X.mul(X);
	Y2 = Y.mul(Y);
	XY = X.mul(Y);

	Mat gauss = (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16;
	//cout << "gauss=" << gauss << endl;
	Mat A, B, C;
	filter2D(X2, A, CV_32FC1, gauss);
	filter2D(Y2, B, CV_32FC1, gauss);
	filter2D(XY, C, CV_32FC1, gauss);

	//计算角点相应R
	float row1 = gray1.rows;
	float col1 = gray1.cols;

	Mat det_M, trace_M, R;
	det_M = A.mul(B) - C.mul(C);
	trace_M = A + B;
	R = det_M - 0.04 * trace_M.mul(trace_M);
	Mat R1 = trace_M.mul(trace_M) / (det_M + 0.001); //保证除数不为0
	//找角点
	float max_R = *(max_element(R.begin<float>(), R.end<float>()));
	//cout << "max_R=" << max_R << endl;
	Mat temp;
	float max_temp;
	Point point;//opencv提供的 用结构体表示的坐标
	//找角点
	vector<float> pix; //中间变量
	vector<vector<float>> pix_corner; //记录角点坐标
	for (int i = 0; i < row1 - 2; i++) {
		for (int j = 0; j < col1 - 2; j++) {
			temp = R(Rect(j, i, 3, 3));//滑动窗口大小
			max_temp = *(max_element(temp.begin<float>(), temp.end<float>()));
			if (max_temp >= 0.03 * max_R && R.at<float>(i, j) == max_temp && R1.at<float>(i, j) > 2 && R1.at<float>(i, j) < 7) { //大于阈值(k*max_R)认为是角点 
				point.y = i; //给坐标point赋值
				point.x = j;
				pix.push_back(i);
				pix.push_back(j);
				pix_corner.push_back(pix);//记录了坐标
				pix.clear(); //保证每一轮循环，pix向量都是空值
				circle(img1, point, 1, Scalar(0, 255, 255));
			}
		}
	}
	for (vector<vector<float>>::iterator it = pix_corner.begin(); it != pix_corner.end(); it++) {
		vector<float>::iterator iti = (*it).begin();//每行的首地址
		float x = *iti; // 行坐标AAA
		iti++;
		float y = *iti; //列坐标
		if (y - 10 > 0 && x - 10 > 0 && y + 10 < col1 && x + 10 < row1) { //满足边界条件
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
	//使用canny 二值化
	Mat canny_img;
	//canny输出的图像为CV_8UC1需转化为CV_32FC1
	Canny(img1, canny_img, 60, 120, 3);
	canny_img.convertTo(canny_img, CV_32FC1);
	//3为sobel滤波模板大小
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
					// 投票
					hough_mat.at<float>(round(temp) + row1, theta)++;
				}
			}
		}
	}
	//创建存储直线参数的二维向量
	vector<Vec2f> lines;
	//等价于vector<vector<float>> lines;
	vector<Vec2f> lines2;
	//myHoughLines(img1, 1, 1, 100, lines2);
	double max_judge = find_max(hough_mat);
	int judge1 = max_judge * 0.19;
	int judge2 = max_judge * 0.195;
	//遍历hongh矩阵
	for (int i = 0; i < 2 * row1 - 2; i++) {
		for (int j = 0; j < 180 - 2; j++) {
			if (hough_mat.at<float>(i, j) > judge1 && hough_mat.at<float>(i, j) < judge2) {
				//保存大于阈值的rou
				Vec2f point(i, j * PI / 180);
				//滤波处理3*3
				hough_mat.at<float>(i, j + 1) = 0;
				hough_mat.at<float>(i + 1, j) = 0;
				hough_mat.at<float>(i + 1, j + 1) = 0;
				hough_mat.at<float>(i + 1, j + 2) = 0;
				hough_mat.at<float>(i + 2, j) = 0;
				hough_mat.at<float>(i + 2, j + 1) = 0;
				hough_mat.at<float>(i + 2, j + 2) = 0;
				lines.push_back(point);
				//将i与j的弧度制存放
			}
		}
	}
	//cout << "line1(滤波处理后)" << endl;

	//使用line1绘制图像
	for (int i = 0; i < size(lines); i++) {
		draw_line(img1, lines[i][0], lines[i][1], row1);
	}
	return img1;
}



int main() {
	Mat img = imread("D:/桌面文件夹/学习资料/图像分析与理解/week_1/上课材料/6.jpg");
	Mat img1 = Corner_detection(img);
	imshow("角点检测", img1);
	Mat img2 = Hough_straight_line(img1);
	imshow("角点检测+hough", img2);
	waitKey(0);
	return 0;
}