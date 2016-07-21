#include <iostream>
#include "Geometry.hpp"
#include "EquiRecFeatureMatching.hpp"

void test_EquiRecFeatureMatching();

void test_EquiRecFeatureMatching() {
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

	// Scale image
	if (inImg1.size().width > 1600) {
		int H = inImg1.size().height;
		int W = inImg1.size().width;
		float scale = 1600. / W;
		H = int(H * scale);
		W = int(W * scale);
		cv::resize(inImg1, inImg1, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
		cv::resize(inImg2, inImg2, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
	}

	// Scalled siize
	int H = inImg1.size().height;
	int W = inImg1.size().width;
	std::cout <<  "Scalled size " << inImg1.size() << std::endl;

	EquiRecFeatureMatching efm(inImg1, inImg2);

	cv::Mat img;
	cv::drawMatches(inImg1, efm.keyPoints1, inImg2, efm.keyPoints2, efm.goodMatches, img);
	cv::imshow("Viewer", img);

	std::cout << "done" << std::endl;

	while (true) {
		if ((cv::waitKey(0) & 255) == 27) {
			break;
		}
	}
}

int main (int argc, char *argv[]) {
	test_EquiRecFeatureMatching();
	return 0;
}
