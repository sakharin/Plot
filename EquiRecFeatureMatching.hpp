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
		int H, W;
		int previousH;
		int previousW;
		cv::Mat vecs;
		cv::Mat mask;
		Geometry geo;

		void point2Vec(cv::Mat*, cv::Point2f);
		void vec2Point(cv::Point2f*, cv::Mat);

	public:
		EquiRecFeatureMatching();
		void detectAndCompute(cv::Mat, std::vector< cv::KeyPoint >*, cv::Mat*);
};
#endif
