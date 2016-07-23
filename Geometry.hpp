#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifndef PI
#define PI M_PI
#endif

#ifndef TWOPI
#define TWOPI (M_PI * 2)
#endif

#ifndef PIOTWO
#define PIOTWO (M_PI * 0.5)
#endif


class Geometry {
	protected:
		void vecElem2Angs(float*, float*, float, float, float);
		void twoPts2VecPtPt(cv::Mat*, cv::Mat, cv::Mat);
		void twoPts2VecPtMat(cv::Mat*, cv::Mat, cv::Mat);
	public:
		Geometry();
		void getRMatrixEulerAngles(cv::Mat*, float, float, float);
		float angsDiff(float, float);
		float normVec(cv::Mat);
		void normalizedVec(cv::Mat*, cv::Mat);
		void vec2Angs(float *, float *, cv::Mat);
		void vec2Angs(cv::Mat*, cv::Mat*, cv::Mat);
		void angs2Vec(cv::Mat*, float, float);
		void twoPts2Vec(cv::Mat*, cv::Mat, cv::Mat);
		void twoPts2Angs(float*, float*, cv::Mat, cv::Mat);
		void twoPts2Angs(cv::Mat*, cv::Mat*, cv::Mat, cv::Mat);
		void u2Phi(float* phi, float u, int W);
		void v2Theta(float* theta, float v, int H);
		void phi2u(float* u, float phi, int W);
		void theta2v(float* v, float theta, int H);
};
#endif
