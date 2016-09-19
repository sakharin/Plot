#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

#include "Geometry.hpp"

#define H 100
#define W 120

#define compareFloat0(a) (((a) < FLT_EPSILON) && (-(a) > -FLT_EPSILON))

#define NUMTEST 1000

void test_angsDiff();

void test_vec2Angs();
void test_vec2Angs_Vec();
void test_vec2Angs_Mat();

void test_angs2Vec();

void test_twoPts2Vec();
void test_twoPts2Vec_PtPt();
void test_twoPts2Vec_PtMat();

void test_twoPts2Angs();
void test_twoPts2Angs_PtPt();
void test_twoPts2Angs_PtMat();

void test_u2Phi();
void test_v2Theta();
void test_phi2u();
void test_theta2v();

void test_angsDiff() {
	std::cout << "testing angsDiff() ..." << std::flush;
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
		assert(compareFloat0(0.001 * (diff1 - diff2)));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_vec2Angs() {
	std::cout << "testing vec2Angs() ..." << std::flush;
	test_vec2Angs_Vec();
	test_vec2Angs_Mat();
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_vec2Angs_Vec() {
	Geometry geo = Geometry();
	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	float phi, theta;

	vec.at<float>(0, 0) = 0;
	vec.at<float>(1, 0) = 0;
	vec.at<float>(2, 0) = 0;

	geo.vec2Angs(vec, &phi, &theta);

	assert(compareFloat0(phi));
	assert(compareFloat0(theta));

	for(int i = 1; i < 360; i++) {
		for(int j = 1; j < 360; j++) {
			for(int k = 1; k < 360; k++) {
				float x = i / 360.;
				float y = j / 360.;
				float z = k / 360.;

				vec.at<float>(0, 0) = x;
				vec.at<float>(1, 0) = y;
				vec.at<float>(2, 0) = z;

				geo.vec2Angs(vec, &phi, &theta);

				float r = std::sqrt(x * x + y * y + z * z);
				assert(compareFloat0(phi - std::atan2(y, x)));
				assert(compareFloat0(theta - std::acos(z / r)));
			}
		}
	}
}

void test_vec2Angs_Mat() {
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
				assert(compareFloat0(phis.at<float>(i, j)));
				assert(compareFloat0(thetas.at<float>(i, j)));
			} else {
				assert(compareFloat0(phis.at<float>(i, j) - std::atan2(y, x)));
				assert(compareFloat0(thetas.at<float>(i, j) - std::acos(z / r)));
			}
		}
	}
}

void test_angs2Vec() {
	std::cout << "testing angs2Vec() ..." << std::flush;
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

			assert(compareFloat0(vec.at<float>(0, 0) - x));
			assert(compareFloat0(vec.at<float>(1, 0) - y));
			assert(compareFloat0(vec.at<float>(2, 0) - z));
		}
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_twoPts2Vec() {
	std::cout << "testing twoPts2Vec() ..." << std::flush;
	test_twoPts2Vec_PtPt();
	test_twoPts2Vec_PtMat();
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_twoPts2Vec_PtPt() {
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

	assert(compareFloat0(vec.at<float>(0, 0) - x));
	assert(compareFloat0(vec.at<float>(1, 0) - y));
	assert(compareFloat0(vec.at<float>(2, 0) - z));
}

void test_twoPts2Vec_PtMat() {
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

			assert(compareFloat0(*x3 - x));
			assert(compareFloat0(*y3 - y));
			assert(compareFloat0(*z3 - z));

			x2 += 3;
			y2 += 3;
			z2 += 3;

			x3 += 3;
			y3 += 3;
			z3 += 3;
		}
	}
}

void test_twoPts2Angs() {
	std::cout << "testing twoPts2Angs() ..." << std::flush;
	test_twoPts2Angs_PtPt();
	test_twoPts2Angs_PtMat();
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_twoPts2Angs_PtPt() {
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
				assert(compareFloat0(phi));
				assert(compareFloat0(theta));
			} else {
				assert(compareFloat0(phi - std::atan2(y, x)));
				assert(compareFloat0(theta - std::acos(z / r)));
			}
		}
	}
}

void test_twoPts2Angs_PtMat() {
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
				assert(compareFloat0(phis.at<float>(i, j)));
				assert(compareFloat0(thetas.at<float>(i, j)));
			} else {
				assert(compareFloat0(phis.at<float>(i, j) - std::fmod(std::atan2(y, x) + TWOPI, TWOPI)));
				assert(compareFloat0(thetas.at<float>(i, j) - std::acos(z / r)));
			}

		}
	}
}

void test_u2Phi() {
	std::cout << "testing u2Phi() ..." << std::flush;
	Geometry geo = Geometry();
	for(int i = 0; i < W; i++) {
		float u = i;
		float phi;
		geo.u2Phi(u, W, &phi);
		float p = u * -TWOPI / W + TWOPI;
		assert(compareFloat0(phi -p));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_v2Theta() {
	std::cout << "testing v2Theta() ..." << std::flush;
	Geometry geo = Geometry();
	for(int i = 0; i < H; i++) {
		float v = i;
		float theta;
		geo.v2Theta(v, H, &theta);
		float t = v * PI / H;
		assert(compareFloat0(theta - t));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_phi2u() {
	std::cout << "testing phi2u() ..." << std::flush;
	Geometry geo = Geometry();
	for(int i = 0; i < 360; i++) {
		float phi = i / 360. * TWOPI;
		float u;
		geo.phi2u(phi, W, &u);
		float u2 = std::fmod((phi - TWOPI) * W / -TWOPI, W);
		assert(compareFloat0(u - u2));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}
void test_theta2v() {
	std::cout << "testing theta2v() ..." << std::flush;
	Geometry geo = Geometry();
	for(int i = 0; i < 180; i++) {
		float theta = i / 180. * PI;
		float v;
		geo.theta2v(theta, H, &v);
		float v2 = theta * H / PI;
		assert(compareFloat0(v - v2));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_writePts3_readPts3() {
	std::cout << "testing writePts3(), readPts3() ..." << std::flush;
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
		assert(compareFloat0(ptsSrc[i].x - ptsDst[i].x));
		assert(compareFloat0(ptsSrc[i].y - ptsDst[i].y));
		assert(compareFloat0(ptsSrc[i].z - ptsDst[i].z));
	}
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

void test_genPts3Random() {
	std::cout << "testing getPt3Random() ..." << std::flush;
	Geometry geo = Geometry();
	std::vector< cv::Point3d > ptsSrc;
	geo.genPts3Random(ptsSrc);
	std::cout << "\b\b\bdone." << std::endl << std::flush;
}

int main (int argc, char *argv[]) {
	test_angsDiff();
	test_vec2Angs();
	test_angs2Vec();
	test_twoPts2Vec();
	test_twoPts2Angs();
	test_u2Phi();
	test_v2Theta();
	test_phi2u();
	test_theta2v();
	test_writePts3_readPts3();
	test_genPts3Random();
	return 0;
}
