#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>

#include "Geometry.hpp"

#define H 100
#define W 120
#define FLOAT_NEAR 1e-3

#define NUMTEST 1000

TEST(RMatrixEulerAngles, All) {
	Geometry geo = Geometry();
	cv::Mat Xs = cv::Mat(NUMTEST, 1, CV_32F);
	cv::Mat Ys = cv::Mat(NUMTEST, 1, CV_32F);
	cv::Mat Zs = cv::Mat(NUMTEST, 1, CV_32F);
	cv::randu(Xs, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(Ys, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(Zs, cv::Scalar(-1000), cv::Scalar(1000));
	for (int i = 0; i < NUMTEST; i++) {
		float x = Xs.at<float>(i, 0);
		float y = Ys.at<float>(i, 0);
		float z = Zs.at<float>(i, 0);

		cv::Mat M = cv::Mat::zeros(3, 3, CV_32F);
		geo.getRMatrixEulerAngles(x, y, z, &M);

		float est_x, est_y, est_z;
		geo.RMatrix2EulerAngles(M, &est_x, &est_y, &est_z);

		ASSERT_NEAR(geo.constrainAngle02PI(x),
						geo.constrainAngle02PI(est_x),
						FLOAT_NEAR);
		ASSERT_NEAR(geo.constrainAngle02PI(y),
						geo.constrainAngle02PI(est_y),
						FLOAT_NEAR);
		ASSERT_NEAR(geo.constrainAngle02PI(z),
						geo.constrainAngle02PI(est_z),
						FLOAT_NEAR);
  }
}

TEST(angsDiffTest, All) {
	Geometry geo = Geometry();
	cv::Mat ang1 = cv::Mat(NUMTEST, 1, CV_32F);
	cv::Mat ang2 = cv::Mat(NUMTEST, 1, CV_32F);
	cv::randu(ang1, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(ang2, cv::Scalar(-1000), cv::Scalar(1000));
	for (int i = 0; i < NUMTEST; i++) {
		float a = ang1.at<float>(i, 0);
		float b = ang2.at<float>(i, 0);

		float diff1 = geo.angsDiff(a, b);

		a = std::fmod(a, TWOPI);
		b = std::fmod(b, TWOPI);
		float diff2 = a - b;
		if (diff2 > PI) {
			diff2 -= TWOPI;
		}
		if (diff2 < -PI) {
			diff2 += TWOPI;
		}
		diff1 = geo.constrainAngle02PI(diff1);
		diff2 = geo.constrainAngle02PI(diff2);
		ASSERT_NEAR(diff1, diff2, FLOAT_NEAR);
	}
}

TEST(vec2AngTest, Vector) {
	Geometry geo = Geometry();
	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	float phi, theta;

	vec.at<float>(0, 0) = 0;
	vec.at<float>(1, 0) = 0;
	vec.at<float>(2, 0) = 0;

	geo.vec2Angs(vec, &phi, &theta);

	ASSERT_FLOAT_EQ(phi, 0);
	ASSERT_FLOAT_EQ(theta, 0);

	cv::Mat pts = cv::Mat(NUMTEST, 3, CV_32F);
	cv::randu(pts, cv::Scalar(-1000), cv::Scalar(1000));
	for(int i = 0; i < NUMTEST; i++) {
		float x = pts.at< float>(i, 0);
		float y = pts.at< float>(i, 1);
		float z = pts.at< float>(i, 2);

		vec.at<float>(0, 0) = x;
		vec.at<float>(1, 0) = y;
		vec.at<float>(2, 0) = z;

		geo.vec2Angs(vec, &phi, &theta);

		float r = std::sqrt(x * x + y * y + z * z);
		ASSERT_FLOAT_EQ(phi, std::atan2(y, x));
		ASSERT_FLOAT_EQ(theta, std::acos(z / r));
	}
}

TEST(vec2AngTest, Matrix) {
	Geometry geo = Geometry();
	cv::Mat vecs = cv::Mat::zeros(H, W, CV_32FC3);
	cv::Mat phis = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat thetas = cv::Mat::zeros(H, W, CV_32F);
	cv::randu(vecs, cv::Scalar(-1000), cv::Scalar(1000));

	geo.vec2Angs(vecs, &phis, &thetas);

	float *p;
	float x, y, z, r;
	for(int i = 0; i < H; i++) {
		p = vecs.ptr<float>(i);
		for(int j = 0; j < W; j++) {
			x = *p++;
			y = *p++;
			z = *p++;

			r = std::sqrt(x * x + y * y + z * z);
			if(r == 0) {
				ASSERT_FLOAT_EQ(phis.at<float>(i, j), 0);
				ASSERT_FLOAT_EQ(thetas.at<float>(i, j), 0);
			} else {
				ASSERT_FLOAT_EQ(phis.at<float>(i, j), std::atan2(y, x));
				ASSERT_FLOAT_EQ(thetas.at<float>(i, j), std::acos(z / r));
			}
		}
	}
}

TEST(angs2VeTest, All) {
	Geometry geo = Geometry();
	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	float phi, theta;

	for(int i = 0; i < 360; i++) {
		for(int j = 0; j < 180; j++) {
			phi = i / 360. * TWOPI;
			theta = j / 180. * PI;
			geo.angs2Vec(phi, theta, &vec);

			float x = std::sin(theta);
			float z = std::cos(theta);
			float y = x * std::sin(phi);
			x = x * std::cos(phi);

			ASSERT_FLOAT_EQ(vec.at<float>(0, 0), x);
			ASSERT_FLOAT_EQ(vec.at<float>(1, 0), y);
			ASSERT_FLOAT_EQ(vec.at<float>(2, 0), z);
		}
	}
}

TEST(twoPts2VecTest, PointPoint) {
	Geometry geo = Geometry();
	cv::Mat P1 = cv::Mat(3, 1, CV_32F);
	cv::Mat P2 = cv::Mat(3, 1, CV_32F);
	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	cv::randu(P1, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(P2, cv::Scalar(-1000), cv::Scalar(1000));

	geo.twoPts2Vec(P1, P2, &vec);

	float x, y, z, r;
	P1 = P2 - P1;
	x = P1.at<float>(0, 0);
	y = P1.at<float>(1, 0);
	z = P1.at<float>(2, 0);
	r = std::sqrt(x * x + y * y + z * z);
	x /= r;
	y /= r;
	z /= r;

	ASSERT_FLOAT_EQ(vec.at<float>(0, 0), x);
	ASSERT_FLOAT_EQ(vec.at<float>(1, 0), y);
	ASSERT_FLOAT_EQ(vec.at<float>(2, 0), z);
}

TEST(twoPts2VecTest, PointMatrix) {
	Geometry geo = Geometry();
	cv::Mat P1 = cv::Mat(3, 1, CV_32F);
	cv::Mat P2 = cv::Mat(H, W, CV_32FC3);
	cv::Mat vec = cv::Mat::zeros(H, W, CV_32FC3);
	cv::randu(P1, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(P2, cv::Scalar(-1000), cv::Scalar(1000));

	geo.twoPts2Vec(P1, P2, &vec);

	float x1 = P1.at<float>(0, 0);
	float y1 = P1.at<float>(1, 0);
	float z1 = P1.at<float>(2, 0);
	float x, y, z, r;
	float *x2, *y2, *z2;
	float *x3, *y3, *z3;
	for(int i =0; i < H; i++) {
		x2 = P2.ptr<float>(i);
		y2 = x2 + 1;
		z2 = x2 + 2;

		x3 = vec.ptr<float>(i);
		y3 = x3 + 1;
		z3 = x3 + 2;
		for(int j =0; j < W; j++) {
			x = *x2 - x1;
			y = *y2 - y1;
			z = *z2 - z1;
			r = std::sqrt(x * x + y *y + z * z);
			x /= r;
			y /= r;
			z /= r;

			ASSERT_FLOAT_EQ(*x3, x);
			ASSERT_FLOAT_EQ(*y3, y);
			ASSERT_FLOAT_EQ(*z3, z);

			x2 += 3;
			y2 += 3;
			z2 += 3;

			x3 += 3;
			y3 += 3;
			z3 += 3;
		}
	}
}

TEST(twoPts2AngsTest, PointPoint) {
	Geometry geo = Geometry();
	cv::Mat P1 = cv::Mat(3, 1, CV_32F);
	cv::Mat P2 = cv::Mat(3, 1, CV_32F);
	float phi, theta;

	for(int i = 0; i < H; i++) {
		for(int j = 0; j < W; j++) {
			cv::randu(P1, cv::Scalar(-1000), cv::Scalar(1000));
			cv::randu(P2, cv::Scalar(-1000), cv::Scalar(1000));

			geo.twoPts2Angs(P1, P2, &phi, &theta);

			float x, y, z, r;
			P1 = P1 - P2;
			x = P1.at<float>(0, 0);
			y = P1.at<float>(1, 0);
			z = P1.at<float>(2, 0);

			r = std::sqrt(x * x + y * y + z * z);
			if(r == 0) {
				ASSERT_FLOAT_EQ(phi, 0);
				ASSERT_FLOAT_EQ(theta, 0);
			} else {
				ASSERT_FLOAT_EQ(phi, std::atan2(y, x));
				ASSERT_FLOAT_EQ(theta, std::acos(z / r));
			}
		}
	}
}

TEST(twoPts2AngsTest, PointMatrix) {
	Geometry geo = Geometry();
	cv::Mat P1 = cv::Mat(3, 1, CV_32F);
	cv::Mat P2 = cv::Mat(H, W, CV_32FC3);
	cv::Mat thetas = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat phis = cv::Mat::zeros(H, W, CV_32F);
	cv::randu(P1, cv::Scalar(-1000), cv::Scalar(1000));
	cv::randu(P2, cv::Scalar(-1000), cv::Scalar(1000));

	float x1 = P1.at<float>(0, 0);
	float y1 = P1.at<float>(1, 0);
	float z1 = P1.at<float>(2, 0);

	geo.twoPts2Angs(P1, P2, &phis, &thetas);

	float x, y, z, r;
	float *p2;
	for(int i = 0; i < H; i++) {
		p2 = P2.ptr<float>(i);
		for(int j = 0; j < W; j++) {
			x = x1 - *p2++;
			y = y1 - *p2++;
			z = z1 - *p2++;

			r = std::sqrt(x * x + y * y + z * z);
			if(r == 0) {
				ASSERT_FLOAT_EQ(phis.at<float>(i, j), 0);
				ASSERT_FLOAT_EQ(thetas.at<float>(i, j), 0);
			} else {
				ASSERT_FLOAT_EQ(phis.at<float>(i, j), std::atan2(y, x));
				ASSERT_FLOAT_EQ(thetas.at<float>(i, j), std::acos(z / r));
			}

		}
	}
}

TEST(u2PhiTest, All) {
	Geometry geo = Geometry();
	for(int i = 0; i < W; i++) {
		float u = i;
		float phi;
		geo.u2Phi(u, W, &phi);
		float p = u * -TWOPI / W + TWOPI;
		ASSERT_FLOAT_EQ(phi, p);
	}
}

TEST(v2ThetaTest, All) {
	Geometry geo = Geometry();
	for(int i = 0; i < H; i++) {
		float v = i;
		float theta;
		geo.v2Theta(v, H, &theta);
		float t = v * PI / H;
		ASSERT_FLOAT_EQ(theta, t);
	}
}

TEST(phi2uTest, All) {
	Geometry geo = Geometry();
	for(int i = 0; i < 360; i++) {
		float phi = i / 360. * TWOPI;
		float u;
		geo.phi2u(phi, W, &u);
		float u2 = std::fmod((phi - TWOPI) * W / -TWOPI, W);
		ASSERT_FLOAT_EQ(u, u2);
	}
}

TEST(theta2vTest, All) {
	Geometry geo = Geometry();
	for(int i = 0; i < 180; i++) {
		float theta = i / 180. * PI;
		float v;
		geo.theta2v(theta, H, &v);
		float v2 = theta * H / PI;
		ASSERT_FLOAT_EQ(v, v2);
	}
}

TEST(writePts3_readPts3Test, All) {
	Geometry geo = Geometry();

	// Generate points
	std::vector< cv::Point3d > ptsSrc;
	cv::Mat dat = cv::Mat(3, NUMTEST, CV_64F);
	for (int i = 0; i < NUMTEST; i++) {
		cv::randu(dat, cv::Scalar(-1000), cv::Scalar(1000));
		double x = dat.at<double>(0, i);
		double y = dat.at<double>(1, i);
		double z = dat.at<double>(2, i);
		cv::Point3d p(x, y, z);
		ptsSrc.push_back(p);
	}

	// Write file
	std::string fileName = "tmp.csv";
	geo.writePts3(fileName, ptsSrc);

	// Read file
	std::vector< cv::Point3d > ptsDst;
	geo.readPts3(fileName, ptsDst);

	// Compare data
	for (int i = 0; i < NUMTEST; i++) {
		ASSERT_FLOAT_EQ(ptsSrc[i].x, ptsDst[i].x);
		ASSERT_FLOAT_EQ(ptsSrc[i].y, ptsDst[i].y);
		ASSERT_FLOAT_EQ(ptsSrc[i].z, ptsDst[i].z);
	}
}

TEST(genPts3RandomTest, All) {
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3Random(ptsSrc);
}

TEST(genPts3UnitSphereTest, All) {
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3UnitSphere(ptsSrc);
}

TEST(genPts3UnitCylinderTest, All) {
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3UnitCylinder(ptsSrc);
}

TEST(genPts3UnitCubeTest, All) {
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3UnitCube(ptsSrc);
}

TEST(genPts3OrthogonalPlanesTest, All) {
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3OrthogonalPlanes(ptsSrc);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
