
#include <opencv.hpp>
#include "TrackerKCF.h"
#include <iostream>
#include <io.h>
#include <Windows.h>
#include <sstream>

using namespace std;
using namespace cv;


bool select_flag = false;
Point pt_origin;
Point pt_end;

double SimilarityCompare(Mat img1, Mat img2);


double SimilarityCompare(Mat img1, Mat img2)
{
	Mat img1_hsv, img2_hsv;

	cvtColor(img1, img1_hsv, CV_BGR2HSV);
	cvtColor(img2, img2_hsv, CV_BGR2HSV);

	/// ��hueͨ��ʹ��30��bin,��saturatoinͨ��ʹ��32��bin
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue��ȡֵ��Χ��0��256, saturationȡֵ��Χ��0��180
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };

	const float* ranges[] = { h_ranges, s_ranges };

	// ʹ�õ�0�͵�1ͨ��
	int channels[] = { 0, 1 };

	MatND hist_img1;
	MatND hist_img2;

	calcHist(&img1_hsv, 1, channels, Mat(), hist_img1, 2, histSize, ranges, true, false);
	normalize(hist_img1, hist_img1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&img2_hsv, 1, channels, Mat(), hist_img2, 2, histSize, ranges, true, false);
	normalize(hist_img2, hist_img2, 0, 1, NORM_MINMAX, -1, Mat());

	double compare_val = compareHist(hist_img1, hist_img2, CV_COMP_BHATTACHARYYA);
	return compare_val;
}

void help()
{
	cout << "--Choose the video source and put Enter: " << endl;
	cout << "	0: choose video frome file" << endl;
	cout << "	1: choose video frome camera" << endl;
}

void getFiles(string path, vector<string>& files)
{
	//�ļ����
	long hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮
			//�������,�����б�
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/************************************************************************/
/* ������Ƶ��Ϣ����ȡ��ʼ����λ�� */
/************************************************************************/
void loadVideoInfo(vector<string> &imageList, Rect &rect)
{
	ifstream dirlist;
	dirlist.open("list.txt");

	if (!dirlist.is_open())
	{
		cout << "list open failed!" << endl;
		exit(-1);
	}

	int k = 0;
	string dirname;
	vector<string> imageDirList;

	while (!dirlist.eof())
	{
		dirlist >> dirname;
		imageDirList.push_back(dirname);
		cout << k << ":" << dirname << endl;
		k++;
	}

	dirlist.close();

	cout << "choose a video:" << endl;
	k = 0;
	cin >> k;

	dirname = imageDirList[k];

	string basePath("D:\\ѧϰ���\\Track\\TrackData\\Benchmark");
	basePath += "\\";
	basePath += dirname;

	string imagePath = basePath;
	imagePath += "\\img";
	getFiles(imagePath, imageList);

	string initPosPath;
	initPosPath = basePath;
	initPosPath += "\\groundtruth_rect.txt";

	ifstream file1;
	file1.open(initPosPath);

	if (!file1.is_open())
	{
		cout << initPosPath << " open failed!" << endl;
		exit(-1);
	}

	char temp[255];
	int a = 0;
	int b = 0;
	int c = 0;
	int d = 0;

	file1.getline(temp, 255);

	sscanf(temp, "%d %d %d %d", &a, &b, &c, &d);
	if (d == 0)
	{
		sscanf(temp, "%d,%d,%d,%d", &a, &b, &c, &d);
	}

	if (d == 0)
	{
		cout << "read groundtruth failed!" << endl;
		exit(-1);
	}

	rect.x = a;
	rect.y = b;
	rect.width = c;
	rect.height = d;

}


void onMouse(int event, int x, int y, int, void* param)
{
	//Point origin;//����������ط����ж��壬��Ϊ���ǻ�����Ϣ��Ӧ�ĺ�����ִ�����origin���ͷ��ˣ����Դﲻ��Ч����
	if (select_flag)
	{
		((Rect*)(param))->x = MIN(pt_origin.x, x);//��һ��Ҫ����굯��ż�����ο򣬶�Ӧ������갴�¿�ʼ���������ʱ��ʵʱ������ѡ���ο�
		((Rect*)(param))->y = MIN(pt_origin.y, y);
		((Rect*)(param))->width = abs(x - pt_origin.x);//����ο�Ⱥ͸߶�
		((Rect*)(param))->height = abs(y - pt_origin.y);
		//select1 &= Rect(0, 0, frame.cols, frame.rows);//��֤��ѡ���ο�����Ƶ��ʾ����֮��
	}
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		select_flag = true;//��갴�µı�־����ֵ
		pt_origin = Point(x, y);//�������������ǲ�׽���ĵ�
		*(Rect*)(param) = Rect(x, y, 0, 0);//����һ��Ҫ��ʼ������͸�Ϊ(0,0)����Ϊ��opencv��Rect���ο����ڵĵ��ǰ������Ͻ��Ǹ���ģ����ǲ������½��Ǹ���
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		pt_end = Point(x, y);
		select_flag = false;
	}
}

#define USE_CAMERA 0

#if USE_CAMERA

int main(int argc, char** argv)
{
	try
	{
		VideoCapture cap;
		Mat imgSrc;//source image
		Mat imgSrcCopy;//source copy

		ostringstream ostr;

		bool start_track = false;

		TickMeter tm;
		tm.reset();

		cap.open(0);
		waitKey(3000);

		if (!cap.isOpened())
		{
			cout << endl;
			cout << "*********************************************" << endl;
			cout << "--------------��Ƶ��ʧ�ܣ�--------------" << endl;
			cout << "--��ȷ���ļ�·����ȷ�Ұ�װ����Ӧ�Ľ�����--��" << endl;
			cout << "*********************************************" << endl;
			cout << endl;
			return -1;
		}

		// Load the first frame.  
		cap >> imgSrc;
		if (NULL == imgSrc.data)
		{
			cout << endl;
			cout << "*********************************************" << endl;
			cout << "--------------��Ƶ��ȡʧ�ܣ�--------------" << endl;
			cout << "--��ȷ���ļ�·����ȷ�Ұ�װ����Ӧ�Ľ�����--��" << endl;
			cout << "*********************************************" << endl;
			cout << endl;
			return -1;
		}

		imgSrc.copyTo(imgSrcCopy);

		imshow("tracker", imgSrcCopy);

		Rect *pselect = new Rect;
		setMouseCallback("tracker", onMouse, pselect);//mouse interface

		cout << "\n--Choose a object to track use mouse" << endl;
		cout << "--Put 'space' to start track or put 'ESC' to exit" << endl;

		//select object roi
		while (true)
		{
			cap >> imgSrc;

			imgSrc.copyTo(imgSrcCopy);
			cv::rectangle(imgSrcCopy, *pselect, CV_RGB(0, 255, 0), 2, 8, 0);
			imshow("tracker", imgSrcCopy);
			char c = waitKey(16);
			if (c == 32)
			{
				start_track = true;
				break;
			}
			else if (c == 27)
			{
				return 1;
			}
		}

		size_t frame = 1;

		// Gray img of imgSrc
		Mat imgGray;

		// Transform the source bgr image to gray
		cvtColor(imgSrc, imgGray, CV_BGR2GRAY);

		TrackerKCF tracker;
		Rect track_rect;

		tracker.init(imgGray, (*pselect));

		Mat img1 = imgSrc(*(pselect));

		// Run the tracker.  call tracker.update() 
		while (true)
		{
			cap >> imgSrc;

			imgSrc.copyTo(imgSrcCopy);

			if (select_flag)
			{
				start_track = false;
				cv::rectangle(imgSrcCopy, *pselect, CV_RGB(0, 255, 0), 2, 8, 0);
				//imshow("tracker", imgSrcCopy);
			}
			else
			{
				if (start_track)
				{
					cvtColor(imgSrcCopy, imgGray, CV_BGR2GRAY);
					tm.reset();
					tm.start();
					track_rect = tracker.update(imgGray);
					tm.stop();
					cout << tm.getTimeMilli() << endl;
					cv::rectangle(imgSrcCopy, track_rect, CV_RGB(255, 0, 0), 2, 8, 0);
					ostr.clear();
					ostr.str("");
					ostr << tracker.psr;
					putText(imgSrcCopy, ostr.str(), Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 192, 192), 2);

					/*Mat img2 = imgSrc(track_rect);

					ostr.clear();
					ostr.str("");
					ostr << SimilarityCompare(img1, img2);
					putText(imgSrcCopy, ostr.str(), Point(20, 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 192, 192), 2);*/

				}
				else
				{
					cv::rectangle(imgSrcCopy, *pselect, CV_RGB(0, 255, 0), 2, 8, 0);
					char c = waitKey(16);
					if (c == 32)
					{
						start_track = true;
						tracker.init(imgGray, (*pselect));
					}
				}
			}

			imshow("tracker", imgSrcCopy);
			if (waitKey(10) == 27)
			{
				break;
			}
		}
	}
	catch (std::exception &ex)
	{
		cout << ex.what();
		return -1;

	}
}

#else
int main()
{
	try
	{
		Mat imgSrc;//source image
		Mat imgSrcCopy;//source copy

		TickMeter tm;
		tm.reset();

		vector< string > imageList;
		Rect initBox;

		loadVideoInfo(imageList, initBox);

		//size_t frame = 1;

		imgSrc = imread(imageList[0]);
		// Gray img of imgSrc
		Mat imgGray;

		// Transform the source bgr image to gray
		cvtColor(imgSrc, imgGray, CV_BGR2GRAY);

		TrackerKCF tracker;
		Rect track_rect;

		tracker.init(imgGray, initBox);
		tracker.update(imgGray);

		string frame_str;
		ostringstream ostr1;

		// Run the tracker.  call tracker.update() 
		for (size_t frame = 1; frame < imageList.size(); frame++)
		{
			imgSrc = imread(imageList[frame]);
			cvtColor(imgSrc, imgGray, CV_BGR2GRAY);

			tm.reset();
			tm.start();
			rectangle(imgSrc, tracker.update(imgGray), Scalar(192, 192, 0), 2);
			tm.stop();
			cout << tm.getTimeMilli() << endl;
			ostr1.clear();
			ostr1.str("");
			ostr1 << frame;
			putText(imgSrc, ostr1.str(), Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1);
			imshow("tracker", imgSrc);

			if (waitKey(5) == 27)
			{
				break;
			}
		}

		return 0;
	}
	catch (std::exception &ex)
	{
		cout << ex.what();
		return -1;

	}
}

#endif