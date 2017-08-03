#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
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
		void vecElem2Angs(float x, float y, float z, float* phi, float* theta);
		void twoPts2VecPtPt(cv::Mat P1, cv::Mat P2, cv::Mat* P);
		void twoPts2VecPtMat(cv::Mat P1, cv::Mat P2, cv::Mat* P);
	public:
		Geometry();
		float constrainAngle0360(float x);
		float constrainAnglem180180(float x);
		float constrainAngle02PI(float x);
		float constrainAnglemPIPI(float x);
		void getRMatrixEulerAngles(float X, float Y, float Z, cv::Mat* M);
		void RMatrix2EulerAngles(cv::Mat M, float* X, float* Y, float* Z);
		float angsDiff(float ang1, float ang2);
		float normVec(cv::Mat vec);
		void normalizedVec(cv::Mat vec, cv::Mat* outVec);
		void vec2Angs(cv::Mat vec, float* phi, float* theta);
		void vec2Angs(cv::Mat vecs, cv::Mat* phis, cv::Mat* thetas);
		void angs2Vec(float phi, float theta, cv::Mat* vec);
		void twoPts2Vec(cv::Mat P1, cv::Mat P2, cv::Mat *P);
		void twoPts2Angs(cv::Mat P1, cv::Mat P2, float* phi, float* theta);
		void twoPts2Angs(cv::Mat P1, cv::Mat P2, cv::Mat* phis, cv::Mat* thetas);
		void u2Phi(float u, int W, float* phi);
		void v2Theta(float v, int H, float* theta);
		void phi2u(float phi, int W, float* u);
		void theta2v(float theta, int H, float* v);

		void writePts3(std::string fileName, std::vector< cv::Point3d >& pts);
		void writeVector(std::string fileName, std::vector< double >& data);
		void writeVector(std::string fileName, std::vector< std::vector< double > >& data);
		void readPts3(std::string fileName, std::vector< cv::Point3d >& pts);
		void readVector(std::string fileName, std::vector< double >& data);
		void readVector(std::string fileName, std::vector< std::vector< double > >& data);

		void genPts3Random(std::vector< cv::Point3d >& pts, int numPts=100, double minDist=0.1, double maxDist=1);
		void genPts3UnitSphere(std::vector< cv::Point3d >& pts, int numPts=100, double r=1.0);
		void genPts3UnitCylinder(std::vector< cv::Point3d >& pts, int numPts=32, int h=5, double r=1.0);
		void genPts3UnitCube(std::vector< cv::Point3d >& pts, int numPts=10, double scale=1.0);
		void genPts3OrthogonalPlanes(std::vector< cv::Point3d >& pts, int numPts=10, double scale=1.0);
};
#endif
