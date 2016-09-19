#include <iostream>
#include "Geometry.hpp"
#include "EquiRecFeatureDetector.hpp"

void test_EquiRecFeatureDetector();

void test_EquiRecFeatureDetector() {
	std::string path = "/home/orion/works/Resources/ThetaData/Set9CamS";
	std::string imgName1 = path + "/R0013629.JPG";
	std::string imgName2 = path + "/R0013646.JPG";

	cv::Mat inImg1 = cv::imread(imgName1, cv::IMREAD_GRAYSCALE);
	cv::Mat inImg2 = cv::imread(imgName2, cv::IMREAD_GRAYSCALE);

	assert(!inImg1.data);
	assert(!inImg2.data);
	assert(inImg1.type() == CV_8U);
	assert(inImg1.type() == inImg2.type());
	assert(inImg1.size() == inImg2.size());

	std::cout <<  "Images are loaded." << std::endl;
	std::cout <<  "Actual size " << inImg1.size() << std::endl;

	// Resized image for speed
	if (inImg1.size().width > 1600) {
		int H = inImg1.size().height;
		int W = inImg1.size().width;
		float scale = 1600. / W;
		H = int(H * scale);
		W = int(W * scale);
		cv::resize(inImg1, inImg1, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
		cv::resize(inImg2, inImg2, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
	}

	// Resized size
	std::cout << "Scalled size " << inImg1.size() << std::endl;

	std::vector< cv::KeyPoint > keyPoints1;
	std::vector< cv::KeyPoint > keyPoints2;
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	std::vector< cv::DMatch > goodMatches;

	// Call EquiRecFeatureDetector
	std::cout << std::endl << "Call EquiRecFeatureDetector ..." << std::flush;
	EquiRecFeatureDetector efm = EquiRecFeatureDetector();
	efm.detectAndCompute(inImg1, &keyPoints1, &descriptors1);
	efm.detectAndCompute(inImg2, &keyPoints2, &descriptors2);
	std::cout << "\b\b\bdone." << std::endl << std::flush;

	// Match features
	cv::FlannBasedMatcher matcher;
	std::vector< std::vector< cv::DMatch > > matches;
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);

	// Ratio test as per Lowe's paper
	for (unsigned int i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
			cv::DMatch m = matches[i][0];
			goodMatches.push_back(m);
		}
	}

	cv::Mat img;
	cv::drawMatches(inImg1, keyPoints1, inImg2, keyPoints2, goodMatches, img);
	cv::imshow("Viewer", img);

	std::cout << "done" << std::endl;

	while (true) {
		if ((cv::waitKey(0) & 255) == 27) {
			break;
		}
	}
}

int main (int argc, char *argv[]) {
	test_EquiRecFeatureDetector();
	return 0;
}
