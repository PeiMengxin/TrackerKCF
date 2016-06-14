#include "trackerKCF.h"

using namespace std;
using namespace cv;

TrackerKCF::TrackerKCF()
{
	psr = 0.0;
}
TrackerKCF::~TrackerKCF()
{
}

#define USE_MERGE 0

/************************************************************************/
/* 初始化跟踪器参数 */
/************************************************************************/

bool TrackerKCF::init(cv::Mat &image, cv::Rect &box)
{
	CV_Assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);
	CV_Assert(box.area() > 100);

	peakSize = Size(9, 9);

	kernel.type = Kernel::GAUSSIAN;
	feature.type = Feature::HOG;

	interp_factor = 0.02;

	kernel.sigma = 0.5;
	kernel.poly_a = 1;
	kernel.poly_b = 9;

	feature.hog_orientations = 9;
	cell_size = 4;
	padding = 2.5;
	output_sigma_factor = 0.1;
	lambda = 0.0001;

	frame = 0;
	MinVal = 0.0;
	MaxVal = 0.0;
	MinLoc = Point(0, 0);
	MaxLoc = Point(0, 0);

	frame = 0;
	roi = box;
	boundingBox = box;
	pos = Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);

	targetSize.width = boundingBox.width;
	targetSize.height = boundingBox.height;

	isResize = false;

	if (roi.area() > 10000)
	{
		isResize = true;

		targetSize.width /= 2.0;
		targetSize.height /= 2.0;

		roi.x /= 2.0;
		roi.y /= 2.0;
		roi.width /= 2.0;
		roi.height /= 2.0;
	}

	// add padding to the roi
	roi.x -= (roi.width / 2 * (padding - 1));
	roi.y -= (roi.height / 2 * (padding - 1));
	roi.width *= padding;
	roi.height *= padding;

	initHOG();

	double output_sigma = sqrtf((float)targetSize.area()) * output_sigma_factor / cell_size;
	Mat label = gaussianLabel(output_sigma, labelSize);
	dft(label, yf, DFT_COMPLEX_OUTPUT);

	gemm(hann(yf.rows), hann(yf.cols), 1, 0, 0, cos_window, GEMM_2_T);

	return true;
}

cv::Rect TrackerKCF::update(cv::Mat &image)
{
	CV_Assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);

	Mat img;

	if (image.type() == CV_8UC3)
	{
		cvtColor(image, img, CV_BGR2GRAY);
	}
	else
	{
		img = image.clone();
	}

	// check the channels of the input image, grayscale is preferred
	CV_Assert(img.channels() == 1 || img.channels() == 3);

	// resize the image whenever needed
	if (isResize)
	{
		resize(img, img, Size(img.cols / 2, img.rows / 2));
	}

	if (frame > 0)
	{
		detect(img);

		boundingBox.x = pos.x - boundingBox.width / 2;
		boundingBox.y = pos.y - boundingBox.height / 2;

		getPSR();
	}

	train(img);

	positions.push_back(pos);

	frame++;

	return boundingBox;
}


void TrackerKCF::initHOG()
{
	hogDescriptor.cellSize = Size(cell_size, cell_size);
	hogDescriptor.blockSize = Size(cell_size, cell_size);
	hogDescriptor.blockStride = Size(cell_size, cell_size);

	hogDescriptor.nbins = 18;

	hogFeature.blockFeatureNumber = hogDescriptor.nbins*(hogDescriptor.blockSize.width / hogDescriptor.cellSize.width)*(hogDescriptor.blockSize.height / hogDescriptor.cellSize.height);

	hogFeature.featureMat_Rows = (roi.height - hogDescriptor.blockSize.height) / hogDescriptor.blockStride.height + 1;
	hogFeature.featureMat_Cols = (roi.width - hogDescriptor.blockSize.width) / hogDescriptor.blockStride.width + 1;

	hogFeature.featureMat_Rows = getOptimalDFTSize(hogFeature.featureMat_Rows);
	hogFeature.featureMat_Cols = getOptimalDFTSize(hogFeature.featureMat_Cols);

	hogDescriptor.winSize.width = (hogFeature.featureMat_Cols - 1)*hogDescriptor.blockStride.width + hogDescriptor.blockSize.width;
	hogDescriptor.winSize.height = (hogFeature.featureMat_Rows - 1)*hogDescriptor.blockStride.height + hogDescriptor.blockSize.height;

	roi.width = hogDescriptor.winSize.width;
	roi.height = hogDescriptor.winSize.height;

	labelSize.width = hogFeature.featureMat_Cols;
	labelSize.height = hogFeature.featureMat_Rows;

}

/************************************************************************/
/* 在线训练 */
/************************************************************************/
void TrackerKCF::train(cv::Mat &img)
{
	Mat patch = getPatch(img);
	Mat temp;

	getHOGFeature(patch);

	if (xfMat.size() > 0)
	{
		xfMat.clear();
	}
	for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
	{
		cv::Mat hog_dft;
		dft(hogMat[k], hog_dft, DFT_COMPLEX_OUTPUT);
		xfMat.push_back(hog_dft);
	}

	temp = kernelCorrelate(xfMat, xfMat);

	static Mat temp_v[2];
	split(temp, temp_v);

	temp_v[0].copyTo(kxxf);

	static Mat yf_v[2];
	split(yf, yf_v);

	static Mat alphaf_v[2];
	alphaf_v[0] = yf_v[0].mul(1 / (kxxf + lambda));
	alphaf_v[1] = yf_v[1].mul(1 / (kxxf + lambda));

	static Mat alphaf;
	merge(alphaf_v, 2, alphaf);

	if (frame == 0)
	{
		alphaf.copyTo(model_alphaf);


		if (model_xfMat.size() > 0)
		{
			model_xfMat.clear();
		}
		for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
		{
			Mat ttt;
			xfMat[k].copyTo(ttt);
			model_xfMat.push_back(ttt);
		}
	}
	else
	{
		model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;

		for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
		{
			model_xfMat[k] = (1 - interp_factor) * model_xfMat[k] + interp_factor * xfMat[k];
		}
	}
}


/************************************************************************/
/* 检测目标 */
/************************************************************************/
void TrackerKCF::detect(cv::Mat &img)
{
	Mat patch = getPatch(img);


	getHOGFeature(patch);

	if (zfMat.size() > 0)
	{
		zfMat.clear();
	}

	for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
	{
		cv::Mat hog_dft;
		dft(hogMat[k], hog_dft, DFT_COMPLEX_OUTPUT);
		zfMat.push_back(hog_dft);
	}

	kzxf = kernelCorrelate(zfMat, model_xfMat);

	mulSpectrums(model_alphaf, kzxf, response_f, DFT_COMPLEX_OUTPUT, false);

	dft(response_f, response, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);

	minMaxLoc(response, &MinVal, &MaxVal, &MinLoc, &MaxLoc);

	if (MaxLoc.x > response.cols / 2)
	{
		MaxLoc.x = MaxLoc.x - response.cols;
	}
	if (MaxLoc.y > response.rows / 2)
	{
		MaxLoc.y = MaxLoc.y - response.rows;
	}
	roi.x += MaxLoc.x*cell_size;
	roi.y += MaxLoc.y*cell_size;

	if (isResize)
	{
		pos.x += 2 * MaxLoc.x*cell_size;
		pos.y += 2 * MaxLoc.y*cell_size;
	}
	else
	{
		pos.x += MaxLoc.x*cell_size;
		pos.y += MaxLoc.y*cell_size;
	}

}

cv::Mat TrackerKCF::getPatch(cv::Mat &img)
{
	Mat dst(roi.size(), img.type());

	int *xs = new int[roi.width];
	int *ys = new int[roi.height];

	int w = img.cols;
	int h = img.rows;

	for (int i = 0; i < roi.height; i++)
	{
		ys[i] = roi.y + i;
		if (ys[i] < 0)
		{
			ys[i] = 0;
		}
		if (ys[i] >= h)
		{
			ys[i] = h - 1;
		}
	}

	for (int i = 0; i < roi.width; i++)
	{
		xs[i] = roi.x + i;
		if (xs[i] < 0)
		{
			xs[i] = 0;
		}
		if (xs[i] >= w)
		{
			xs[i] = w - 1;
		}
	}

	if (dst.type() == CV_8U)
	{
		for (int i = 0; i < roi.width; i++)
		{
			for (int j = 0; j < roi.height; j++)
			{
				dst.at<uchar>(j, i) = img.at<uchar>(ys[j], xs[i]);
			}
		}
	}

	else if (dst.type() == CV_8UC3)
	{
		for (int i = 0; i < roi.width; i++)
		{
			for (int j = 0; j < roi.height; j++)
			{
				dst.at<Vec3b>(j, i) = img.at<Vec3b>(ys[j], xs[i]);
			}
		}
	}

	return dst;
}

cv::Mat TrackerKCF::getFeature(cv::Mat img)
{
	Mat dst;
	img.copyTo(dst);

	switch (feature.type)
	{
		case Feature::HOG:
		{
			getHOGFeature(img);
			break;
		}
		case Feature::GRAY:
		{
			dst.convertTo(dst, CV_32FC1);
			dst = dst / 255;
			dst = dst - mean(dst).val[0];

			if (cos_window.data)
			{
				dst = cos_window.mul(dst);
			}
			break;
		}
		default:
			break;
	}

	return dst;
}

void TrackerKCF::getHOGFeature(cv::Mat img)
{
	hogDescriptor.compute(img, hogFeature.feature_descriptors);

	if (hogMat.size() > 0)
	{
		hogMat.clear();
	}

	Mat feature_mat(hogFeature.featureMat_Rows, hogFeature.featureMat_Cols, CV_32FC1);

	int n1 = hogFeature.featureMat_Rows*hogFeature.blockFeatureNumber;

	for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
	{
		Mat feature_mat(hogFeature.featureMat_Rows, hogFeature.featureMat_Cols, CV_32FC1);
		for (int j = 0; j < hogFeature.featureMat_Cols; j++)
		{
			for (int i = 0; i < hogFeature.featureMat_Rows; i++)
			{
				feature_mat.at<float>(i, j) = hogFeature.feature_descriptors[j*n1 + i*hogFeature.blockFeatureNumber + k];
			}
		}
		hogMat.push_back(feature_mat);
	}

	if (cos_window.data)
	{
		for (int k = 0; k < hogFeature.blockFeatureNumber; k++)
		{
			hogMat[k] = cos_window.mul(hogMat[k]);
		}
	}
}

cv::Mat TrackerKCF::kernelCorrelate(cv::Mat m1, cv::Mat m2)
{
	Mat dst;

	switch (kernel.type)
	{
		case Kernel::GAUSSIAN:
		{
			dst = gaussianCorrelation(m1, m2);
			break;
		}
		case Kernel::POLY:
		{
			dst = polyCorrelation(m1, m2);
			break;
		}
		case Kernel::LINEAR:
		{
			dst = linearCorrelation(m1, m2);
			break;
		}
		default:
			break;
	}

	if (dst.type() != CV_32FC1)
	{
		dst.convertTo(dst, CV_32FC1);
	}

	return dst;
}

cv::Mat TrackerKCF::kernelCorrelate(std::vector<Mat> m1, std::vector<Mat> m2)
{
	Mat dst;

	switch (kernel.type)
	{
		case Kernel::GAUSSIAN:
		{
			dst = gaussianCorrelation(m1, m2);
			break;
		}
		case Kernel::POLY:
		{
			dst = polyCorrelation(m1, m2);
			break;
		}
		case Kernel::LINEAR:
		{
			dst = linearCorrelation(m1, m2);
			break;
		}
		default:
			break;
	}

	if (dst.type() != CV_32FC1)
	{
		dst.convertTo(dst, CV_32FC1);
	}

	return dst;
}

cv::Mat TrackerKCF::gaussianCorrelation(cv::Mat m1, cv::Mat m2)
{

	int N = m1.cols*m1.rows;

	double mm1 = m1.dot(m1) / N;
	double mm2 = m2.dot(m2) / N;

	static Mat m12f;
	mulSpectrums(m1, m2, m12f, DFT_COMPLEX_OUTPUT, true);

	static Mat m12;
	dft(m12f, m12, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);

	Mat dsttemp;
	dsttemp = cv::max<double>((mm1 + mm2 - 2 * m12) / m1.total(), 0);

	exp(dsttemp*(-1 / (kernel.sigma*kernel.sigma)), dsttemp);

	Mat dst;
	dft(dsttemp, dst, DFT_COMPLEX_OUTPUT);

	return dst;
}

cv::Mat TrackerKCF::gaussianCorrelation(std::vector<Mat> m1, std::vector<Mat> m2)
{
	int N = m1[0].cols*m1[0].rows;
	double mm1 = 0;
	double mm2 = 0;
	static Mat m12f;
	mulSpectrums(m1[0], m2[0], m12f, DFT_COMPLEX_OUTPUT, true);

	static Mat m12;
	Mat m12s(m1[0].rows, m1[0].cols, CV_32FC1, Scalar(0));

	for (int i = 0; i < hogFeature.blockFeatureNumber; i++)
	{
		mm1 += m1[i].dot(m1[i]);
		mm2 += m2[i].dot(m2[i]);

		mulSpectrums(m1[i], m2[i], m12f, DFT_COMPLEX_OUTPUT, true);

		dft(m12f, m12, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);
		m12s += m12;
	}

	mm1 = mm1 / N;
	mm2 = mm2 / N;

	Mat dsttemp;
	dsttemp = cv::max<double>((mm1 + mm2 - 2 * m12s) / (m1.size()*N), 0);

	exp(dsttemp*(-1 / kernel.sigma / kernel.sigma), dsttemp);

	Mat dst;
	dft(dsttemp, dst, DFT_COMPLEX_OUTPUT);

	return dst;
}

cv::Mat TrackerKCF::polyCorrelation(cv::Mat m1, cv::Mat m2)
{

	static Mat m12f;
	mulSpectrums(m1, m2, m12f, DFT_COMPLEX_OUTPUT, true);

	static Mat m12;
	dft(m12f, m12, DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE);

	Mat dsttemp;
	pow(m12 / (m1.rows*m1.cols) + kernel.poly_a, kernel.poly_b, dsttemp);

	Mat dst;
	dft(dsttemp, dst, DFT_COMPLEX_OUTPUT);

	return dst;
}

cv::Mat TrackerKCF::polyCorrelation(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2)
{
	Mat dst;

	return dst;
}

cv::Mat TrackerKCF::linearCorrelation(cv::Mat m1, cv::Mat m2)
{
	Mat dst;
	mulSpectrums(m1, m2, dst, DFT_COMPLEX_OUTPUT, true);
	dst = dst / (m1.rows*m1.cols);

	return dst;
}

cv::Mat TrackerKCF::linearCorrelation(std::vector<cv::Mat> m1, std::vector<cv::Mat> m2)
{
	static int N = m1[0].cols*m1[0].rows;
	static Mat m12f(m1[0].rows, m1[0].cols, CV_32FC2, Scalar(0));
	Mat m12s(m1[0].rows, m1[0].cols, CV_32FC2, Scalar(0));

	for (int i = 0; i < hogFeature.blockFeatureNumber; i++)
	{
		mulSpectrums(m1[i], m2[i], m12f, DFT_COMPLEX_OUTPUT, true);
		m12s += m12f;
	}

	Mat dsttemp;
	m12s = m12s / (N*m1.size());
	m12s.copyTo(dsttemp);

	return dsttemp;
}

void TrackerKCF::getPSR()
{
	Mat response_temp = response.clone();

	Mat temp;
	cyclicShift(response_temp, temp, SHIFTTYPE::SHIFTTYPE_DOWN, response_temp.rows / 2);
	cyclicShift(temp, response_temp, SHIFTTYPE::SHIFTTYPE_RIGHT, response_temp.cols / 2);

	Rect peak_rect(response_temp.cols / 2 - peakSize.width / 2, response_temp.rows / 2 - peakSize.height / 2, peakSize.width, peakSize.height);

	Mat peak = response_temp(peak_rect);
	Mat mask(response_temp.size(), CV_8UC1, Scalar(255));

	mask(peak_rect).setTo(0);

	double g_max = MaxVal;
	Scalar mean;
	Scalar stddev;
	meanStdDev(response_temp, mean, stddev, mask);

	psr = (g_max - mean.val[0]) / stddev.val[0];

}

//************************************
// Method:    gaussianLabel
// FullName:  gaussianLabel
// Access:    public 
// Returns:   cv::Mat
// Qualifier:
// Parameter: double output_sigma
// Parameter: cv::Size sz
//************************************
cv::Mat gaussianLabel(double output_sigma, cv::Size sz)
{
	Mat label;
	Mat rs(sz, CV_32FC1);
	Mat cs(sz, CV_32FC1);

	for (int i = 0; i < sz.width; i++)
	{
		cs.at<float>(0, i) = (float)(i - sz.width / 2);
	}
	for (int i = 0; i < sz.height; i++)
	{
		rs.at<float>(i, 0) = (float)(i - sz.height / 2);
	}

	for (int i = 1; i < sz.height; i++)
	{
		cs.row(0).copyTo(cs.row(i));
	}
	for (int i = 1; i < sz.width; i++)
	{
		rs.col(0).copyTo(rs.col(i));
	}

	exp(-0.5 / pow(output_sigma, 2)*(rs.mul(rs) + cs.mul(cs)), label);

	Mat labelTemp;
	cyclicShift(label, labelTemp, SHIFTTYPE_LEFT, sz.width / 2);
	cyclicShift(labelTemp, label, SHIFTTYPE_UP, sz.height / 2);

	CV_Assert(label.at<float>(0, 0) == 1);

	return label;
}



/************************************************************************/
/* 矩阵循环移位 */
/************************************************************************/
void cyclicShift(cv::Mat src, cv::Mat& dst, SHIFTTYPE type, int times)
{

	dst.create(src.size(), src.type());

	switch (type)
	{
		case SHIFTTYPE_LEFT:
		{
			for (int j = 0; j < src.cols; j++)
			{
				src.col((j + times) % src.cols).copyTo(dst.col(j));
			}

			break;
		}

		case SHIFTTYPE_RIGHT:
		{
			for (int j = 0; j < src.cols; j++)
			{
				src.col(j).copyTo(dst.col((j + times) % src.cols));
			}

			break;
		}

		case SHIFTTYPE_UP:
		{
			for (int i = 0; i < src.rows; i++)
			{
				src.row((i + times) % src.rows).copyTo(dst.row(i));
			}

			break;
		}

		case SHIFTTYPE_DOWN:
		{
			for (int i = 0; i < src.rows; i++)
			{
				src.row(i).copyTo(dst.row((i + times) % src.rows));
			}

			break;
		}
		default:
			break;
	}
}

cv::Mat hann(int L)
{
	CV_Assert(L > 1);

	Mat dst(L, 1, CV_32FC1);

	for (int i = 0; i < dst.rows; i++)
	{
		dst.at<float>(i, 0) = float(0.5*(1 - cos(2 * CV_PI*i / (L - 1))));
	}

	return dst;

}
