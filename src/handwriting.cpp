#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

string target_pathName = "9.png"; // enter the name of the picture
string pathName = "database.png";

int SZ = 20;
float affineFlags = WARP_INVERSE_MAP | INTER_LINEAR;

Mat deskew(Mat& img) {
	Moments m = moments(img);
	if (abs(m.mu02) < 1e-2) {
		return img.clone();
	}
	float skew = m.mu11 / m.mu02;
	Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5 * SZ * skew, 0, 1, 0);
	Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
	warpAffine(img, imgOut, warpMat, imgOut.size(), affineFlags);

	return imgOut;
}

void loadTrainTestLabel(string &pathName, string&target_pathName,
		vector<Mat> &trainCells, vector<Mat> &testCells,
		vector<int> &trainLabels) {

	Mat img = imread(pathName, CV_LOAD_IMAGE_GRAYSCALE);

	// process the target image
	Mat img2 = imread(target_pathName, CV_LOAD_IMAGE_GRAYSCALE);
	resize(img2, img2, Size(SZ, SZ), 0, 0, INTER_NEAREST);
	Mat target_img;
	adaptiveThreshold(img2, target_img,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,5,2);

	imshow("asdas",target_img);
	//waitKey();

	testCells.push_back(target_img);
	int ImgCount = 0;
	for (int i = 0; i < img.rows; i = i + SZ) {
		for (int j = 0; j < img.cols; j = j + SZ) {
			ImgCount++;
			if (ImgCount > 500) {
				Mat digitImg =
						(img.colRange(j, j + SZ).rowRange(i, i + SZ)).clone();
				trainCells.push_back(digitImg);
			}
		}
	}

	float digitClassNumber = 1;
	for (int z = 501; z <= ImgCount; z++) {
		if (z % 500 == 0) {
			digitClassNumber++;
		}
		trainLabels.push_back(digitClassNumber);
	}
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,
		vector<Mat> &deskewedTestCells, vector<Mat> &trainCells,
		vector<Mat> &testCells) {

	for (int i = 0; i < trainCells.size(); i++) {

		Mat deskewedImg = deskew(trainCells[i]);
		deskewedTrainCells.push_back(deskewedImg);
	}

	for (int i = 0; i < testCells.size(); i++) {

		Mat deskewedImg = deskew(testCells[i]);
		deskewedTestCells.push_back(deskewedImg);
	}
}

HOGDescriptor hog(Size(20, 20), //winSize
Size(8, 8), //blocksize
Size(4, 4), //blockStride,
Size(8, 8), //cellSize,
9, //nbins,
		1, -1, 0, 0.2, 0, 64, 1);

void CreateTrainTestHOG(vector<vector<float> > &trainHOG,
		vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells,
		vector<Mat> &deskewedtestCells) {

	for (int y = 0; y < deskewedtrainCells.size(); y++) {
		vector<float> descriptors;
		hog.compute(deskewedtrainCells[y], descriptors);
		trainHOG.push_back(descriptors);
	}

	for (int y = 0; y < deskewedtestCells.size(); y++) {

		vector<float> descriptors;
		hog.compute(deskewedtestCells[y], descriptors);
		testHOG.push_back(descriptors);
	}
}
void ConvertVectortoMatrix(vector<vector<float> > &trainHOG,
		vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat) {

	int descriptor_size = trainHOG[0].size();

	for (int i = 0; i < trainHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			trainMat.at<float>(i, j) = trainHOG[i][j];
		}
	}
	for (int i = 0; i < testHOG.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			testMat.at<float>(i, j) = testHOG[i][j];
		}
	}
}

Ptr<SVM> svmInit(float C, float gamma) {

	Ptr<SVM> svm = SVM::create();
	svm->setGamma(gamma);
	svm->setC(C);
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);

	return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels) {
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
	svm->train(td);
	svm->save("eyeGlassClassifierModel.txt");
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat) {
	svm->predict(testMat, testResponse);
}

void svmOutput(Mat &testResponse) {
	for (int i = 0; i < testResponse.rows; i++) {
		cout << testResponse.at<float>(i, 0) << " " << endl;
	}
}

int main() {
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	loadTrainTestLabel(pathName, target_pathName, trainCells, testCells,
			trainLabels);

	vector<Mat> deskewedTrainCells;
	vector<Mat> deskewedTestCells;
	CreateDeskewedTrainTest(deskewedTrainCells, deskewedTestCells, trainCells,
			testCells);

	std::vector<std::vector<float> > trainHOG;
	std::vector<std::vector<float> > testHOG;
	CreateTrainTestHOG(trainHOG, testHOG, deskewedTrainCells,
			deskewedTestCells);

	int descriptor_size = trainHOG[0].size();

	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
	Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

	ConvertVectortoMatrix(trainHOG, testHOG, trainMat, testMat);

	float C = 12.5, gamma = 0.5;

	Mat testResponse;
	Ptr<SVM> model = svmInit(C, gamma);

	svmTrain(model, trainMat, trainLabels);
	svmPredict(model, testResponse, testMat);

	// final output
	svmOutput(testResponse);

	return 0;
}
