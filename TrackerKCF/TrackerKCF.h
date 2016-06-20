#pragma once

#include <opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "piotr_fhog/fhog.hpp"

struct Kernel
{
	enum KernelType
	{
		GAUSSIAN = 1,
		POLY,
		LINEAR
	};

	int type;
	double sigma;
	double poly_a;
	double poly_b;
};

struct Feature
{
	enum FeatureType
	{
		HOG = 1,
		FHOG,
		GRAY
	};

	int type;
	double hog_orientations;
};

struct HOGFeature
{
	int blockFeatureNumber;
	int featureMat_Rows;
	int featureMat_Cols;
	std::vector<float> feature_descriptors;
};

class TrackerKCF
{
public:
	TrackerKCF();
	~TrackerKCF();

	std::vector<cv::Point> positions;
	double psr;

	bool init(cv::Mat &image, cv::Rect &box);
	cv::Rect update(cv::Mat &image);
	cv::Rect getBoundingBox(){ return boundingBox; }
	cv::Point getPosition(){ return pos; }

private:
	
	Kernel kernel;
	Feature feature;

	cv::Rect roi;
	cv::Rect boundingBox;
	cv::Point pos;

	cv::HOGDescriptor hogDescriptor;
	HOGFeature hogFeature;
	std::vector<cv::Mat> hogMat;

	FHoG p_fhog;

	size_t frame;
	bool isResize;

	double interp_factor;
	int cell_size;
	double padding;
	double lambda;
	double output_sigma_factor;

	cv::Size labelSize;
	cv::Size targetSize;
	cv::Size peakSize;

	cv::Mat cos_window;

	cv::Mat xf;
	std::vector<cv::Mat> xfMat;
	cv::Mat yf;
	cv::Mat zf;
	std::vector<cv::Mat> zfMat;

	cv::Mat kzxf;
	cv::Mat kxxf;
	cv::Mat kxxf_real;

	cv::Mat model_alphaf;
	cv::Mat model_alphaf_Num;
	cv::Mat model_alphaf_Den;
	cv::Mat model_xf;
	std::vector<cv::Mat> model_xfMat;

	cv::Mat response;
	cv::Mat response_f;
	cv::Mat response_f_sum;
	cv::Mat K_sum;
	cv::Point MaxLoc;
	cv::Point MinLoc;
	double MaxVal;
	double MinVal;

	void initHOG();

	void train(cv::Mat &img);
	void detect(cv::Mat &img);

	cv::Mat getPatch(cv::Mat &img);
	cv::Mat getFeature(cv::Mat img);
	void getHOGFeature(cv::Mat img);
	void getFHOGFeature(cv::Mat img);

	cv::Mat kernelCorrelate(cv::Mat m1, cv::Mat m2);
	cv::Mat kernelCorrelate(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2);

	cv::Mat gaussianCorrelation(cv::Mat m1, cv::Mat m2);
	cv::Mat gaussianCorrelation(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2);
	cv::Mat polyCorrelation(cv::Mat m1, cv::Mat m2);
	cv::Mat polyCorrelation(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2);
	cv::Mat linearCorrelation(cv::Mat m1, cv::Mat m2);
	cv::Mat linearCorrelation(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2);

	void getPSR();
};

//************************************
// Method:    gaussianLabel
// FullName:  gaussianLabel
// Access:    public 
// Returns:   cv::Mat
// Qualifier:
// Parameter: double output_sigma
// Parameter: cv::Size sz
//************************************
cv::Mat gaussianLabel(double output_sigma, cv::Size sz);

enum SHIFTTYPE
{
	SHIFTTYPE_LEFT = 1,
	SHIFTTYPE_RIGHT,
	SHIFTTYPE_UP,
	SHIFTTYPE_DOWN
};

void cyclicShift(cv::Mat src, cv::Mat& dst, SHIFTTYPE type, int times);

cv::Mat hann(int L);
