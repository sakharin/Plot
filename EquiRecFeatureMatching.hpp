#ifndef __EQUIRECFEATUREMATCHING_H__
#define __EQUIRECFEATUREMATCHING_H__

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "Geometry.hpp"

#ifndef PI
#define PI M_PI
#endif

#ifndef TWOPI
#define TWOPI (M_PI * 2)
#endif


class EquiRecFeatureMatching {
	private:
		int H;
		int W;
		void point2Vec(cv::Mat*, cv::Point2f);
		void vec2Point(cv::Point2f*, cv::Mat);
		Geometry geo;

	public:
		std::vector< cv::KeyPoint > keyPoints1;
		std::vector< cv::KeyPoint > keyPoints2;
		cv::Mat descriptors1;
		cv::Mat descriptors2;
		std::vector< cv::DMatch > goodMatches;

		EquiRecFeatureMatching(cv::Mat inImg1, cv::Mat inImg2);
};
#endif
