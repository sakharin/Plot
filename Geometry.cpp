#include "Geometry.hpp"

Geometry::Geometry() {
}

void Geometry::getRMatrixEulerAngles(float X, float Y, float Z, cv::Mat *M) {
	CV_Assert(M->data != NULL);
	CV_Assert(M->rows == 3 && M->cols == 3);
	CV_Assert(M->type() == CV_32F);

	//Graphic Gems IV, Paul S. Heckbert, 1994
	float sX = sin(X);
	float sY = sin(Y);
	float sZ = sin(Z);

	float cX = cos(X);
	float cY = cos(Y);
	float cZ = cos(Z);

	float rX[] = {1., 0., 0.,
		0., cX, -sX,
		0., sX, cX};
	float rZ[] = {cZ, -sZ, 0.,
		sZ, cZ, 0.,
		0., 0., 1.};
	float rY[] = {cY, 0., sY,
		0., 1., 0.,
		-sY, 0., cY};

	cv::Mat mrX(3, 3, CV_32F, rX);
	cv::Mat mrY(3, 3, CV_32F, rY);
	cv::Mat mrZ(3, 3, CV_32F, rZ);

	cv::Mat tmp = mrX * mrY * mrZ;
	tmp.copyTo(*M);
}

float Geometry::angsDiff(float ang1, float ang2) {
	//http://stackoverflow.com/questions/12234574/calculating-if-an-angle-is-between-two-angles
	// Return diff in range [-pi, pi]
	return std::fmod((ang1 - ang2 + PI), TWOPI) - PI;
}

float Geometry::normVec(cv::Mat vec) {
	CV_Assert(vec.data != NULL);
	CV_Assert(vec.rows == 3 && vec.cols == 1);
	CV_Assert(vec.type() == CV_32F);

	float x = vec.at<float>(0, 0);
	float y = vec.at<float>(1, 0);
	float z = vec.at<float>(2, 0);
	return std::sqrt(x * x + y * y + z * z);
}

void Geometry::normalizedVec(cv::Mat vec, cv::Mat* outVec) {
	CV_Assert(vec.data != NULL);
	CV_Assert(vec.rows == 3 && vec.cols == 1);
	CV_Assert(vec.type() == CV_32F);

	CV_Assert(outVec->data != NULL);
	CV_Assert(outVec->rows == 3 && outVec->cols == 1);
	CV_Assert(outVec->type() == CV_32F);

	*outVec = vec / normVec(vec);
}

void Geometry::vecElem2Angs(float x, float y, float z, float* phi, float* theta) {
	float r;
	r = std::sqrt(x * x + y * y + z * z);
	if(r == 0) {
		*phi = 0;
		*theta = 0;
	} else {
		*phi = std::atan2(y, x);
		*theta = std::acos(z / r);
	}
}

void Geometry::vec2Angs(cv::Mat vec, float* phi, float* theta) {
	CV_Assert(vec.data != NULL);
	CV_Assert(vec.rows == 3 && vec.cols == 1);
	CV_Assert(vec.type() == CV_32F);

	float x, y, z;

	x = vec.at<float>(0, 0);
	y = vec.at<float>(1, 0);
	z = vec.at<float>(2, 0);
	vecElem2Angs(x, y, z, phi, theta);
}

void Geometry::vec2Angs(cv::Mat vecs, cv::Mat* phis, cv::Mat* thetas) {
	CV_Assert(vecs.data != NULL);
	CV_Assert(phis->data != NULL);
	CV_Assert(thetas->data != NULL);

	CV_Assert(vecs.rows == phis->rows && vecs.cols == phis->cols);
	CV_Assert(vecs.rows == thetas->rows && vecs.cols == thetas->cols);

	CV_Assert(vecs.type() == CV_32FC3);
	CV_Assert(phis->type() == CV_32F);
	CV_Assert(thetas->type() == CV_32F);

	int h = vecs.rows;
	int w = vecs.cols;

	if(vecs.isContinuous() && phis->isContinuous() && thetas->isContinuous()) {
		w *= h;
		h = 1;
	}

	int i, j;
	float *x, *y, *z, *p, *t;
	for(i = 0; i < h; i++) {
		x = vecs.ptr<float>(i);
		y = x + 1;
		z = x + 2;

		p = phis->ptr<float>(i);
		t = thetas->ptr<float>(i);
		for(j = 0; j < w; j++) {
			vecElem2Angs(*x, *y, *z, p, t);
			x += 3;
			y += 3;
			z += 3;
			t += 1;
			p += 1;
		}
	}
}

void Geometry::angs2Vec(float phi, float theta, cv::Mat* vec) {
	CV_Assert(vec->data != NULL);
	CV_Assert(vec->rows == 3 && vec->cols == 1);
	CV_Assert(vec->type() == CV_32F);
	CV_Assert(vec->data != NULL);
	vec->at<float>(0, 0) = std::sin(theta);
	vec->at<float>(2, 0) = std::cos(theta);
	vec->at<float>(1, 0) = vec->at<float>(0, 0) * std::sin(phi);
	vec->at<float>(0, 0) = vec->at<float>(0, 0) * std::cos(phi);
}

void Geometry::twoPts2Vec(cv::Mat P1, cv::Mat P2, cv::Mat *P) {
	CV_Assert(P1.data != NULL);
	CV_Assert(P2.data != NULL);
	CV_Assert(P->data != NULL);

	if(P1.rows == 3 && P1.cols == 1 &&
			P2.rows == 3 && P2.cols == 1 &&
			P->rows == 3 && P->cols == 1) {
		CV_Assert(P1.type() == CV_32F);
		CV_Assert(P2.type() == CV_32F);
		CV_Assert(P->type() == CV_32F);
		twoPts2VecPtPt(P1, P2, P);
	} else if(P1.rows == 3 && P1.cols == 1 &&
			P2.rows == P->rows && P2.cols == P->cols) {
		CV_Assert(P1.type() == CV_32F);
		CV_Assert(P2.type() == CV_32FC3);
		CV_Assert(P->type() == CV_32FC3);
		twoPts2VecPtMat(P1, P2, P);
	} else {
		CV_Assert((P1.rows == 3 && P1.cols == 1 &&
					P2.rows == 3 && P2.cols == 1 &&
					P->rows == 3 && P->cols == 1) or
				(P1.rows == 3 && P1.cols == 1 &&
				 P2.rows == P->rows && P2.cols == P->cols));
	}
}

void Geometry::twoPts2VecPtPt(cv::Mat P1, cv::Mat P2, cv::Mat *P) {
	*P = P2 - P1;
	float x = P->at<float>(0, 0);
	float y = P->at<float>(1, 0);
	float z = P->at<float>(2, 0);
	float norm = std::sqrt(x * x + y * y + z * z);
	*P /= norm;
}

void Geometry::twoPts2VecPtMat(cv::Mat P1, cv::Mat P2, cv::Mat *P) {
	int h = P2.rows;
	int w = P2.cols;

	if(P2.isContinuous() && P->isContinuous()) {
		w *= h;
		h = 1;
	}

	int i, j;
	float x1 = P1.at<float>(0, 0);
	float y1 = P1.at<float>(1, 0);
	float z1 = P1.at<float>(2, 0);
	float x, y, z;
	float *x2, *y2, *z2;
	float *x3, *y3, *z3;
	float norm;
	for(i = 0; i < h; i++) {
		x2 = P2.ptr<float>(i);
		y2 = x2 + 1;
		z2 = x2 + 2;

		x3 = P->ptr<float>(i);
		y3 = x3 + 1;
		z3 = x3 + 2;
		for(j = 0; j < w; j++) {
			x = *x2 - x1;
			y = *y2 - y1;
			z = *z2 - z1;
			norm = std::sqrt(x * x + y * y + z * z);
			*x3 = x / norm;
			*y3 = y / norm;
			*z3 = z / norm;

			x2 += 3;
			y2 += 3;
			z2 += 3;

			x3 += 3;
			y3 += 3;
			z3 += 3;
		}
	}
}

void Geometry::twoPts2Angs(cv::Mat P1, cv::Mat P2, float* phi, float* theta) {
	CV_Assert(P1.data != NULL);
	CV_Assert(P1.rows == 3 && P1.cols == 1);
	CV_Assert(P1.type() == CV_32F);

	CV_Assert(P2.data != NULL);
	CV_Assert(P2.rows == 3 && P2.cols == 1);
	CV_Assert(P2.type() == CV_32F);

	float x = P1.at<float>(0, 0) - P2.at<float>(0, 0);
	float y = P1.at<float>(1, 0) - P2.at<float>(1, 0);
	float z = P1.at<float>(2, 0) - P2.at<float>(2, 0);
	vecElem2Angs(x, y, z, phi, theta);
}

void Geometry::twoPts2Angs(cv::Mat P1, cv::Mat P2, cv::Mat* phis, cv::Mat* thetas) {
	CV_Assert(P1.data != NULL);
	CV_Assert(P1.rows == 3 && P1.cols == 1);
	CV_Assert(P1.type() == CV_32F);

	CV_Assert(P2.data != NULL);
	CV_Assert(phis->data != NULL);
	CV_Assert(thetas->data != NULL);

	CV_Assert(P2.rows == phis->rows && P2.cols == phis->cols);
	CV_Assert(P2.rows == thetas->rows && P2.cols == thetas->cols);

	CV_Assert(P2.type() == CV_32FC3);
	CV_Assert(phis->type() == CV_32F);
	CV_Assert(thetas->type() == CV_32F);

	int h = P2.rows;
	int w = P2.cols;

	if(P2.isContinuous() &&
			phis->isContinuous() &&
			thetas->isContinuous()) {
		w *= h;
		h = 1;
	}

	int i, j;
	float x1 = P1.at<float>(0, 0);
	float y1 = P1.at<float>(1, 0);
	float z1 = P1.at<float>(2, 0);
	float x, y, z;
	float *x2, *y2, *z2;
	float *p, *t;
	for(i = 0; i < h; i++) {
		x2 = P2.ptr<float>(i);
		y2 = x2 + 1;
		z2 = x2 + 2;

		p = phis->ptr<float>(i);
		t = thetas->ptr<float>(i);
		for(j = 0; j < w; j++) {
			x = x1 - *x2;
			y = y1 - *y2;
			z = z1 - *z2;

			vecElem2Angs(x, y, z, p++, t++);
			x2 += 3;
			y2 += 3;
			z2 += 3;
		}
	}
}

void Geometry::u2Phi(float u, int W, float* phi) {
	*phi = u * -TWOPI / W + TWOPI;
}

void Geometry::v2Theta(float v, int H, float* theta) {
	*theta = v * PI / H;
}

void Geometry::phi2u(float phi, int W, float* u) {
	*u = std::fmod((phi - TWOPI) * W / -TWOPI, W);
}

void Geometry::theta2v(float theta, int H, float* v) {
	*v = theta * H / PI;
}

void Geometry::writePts3(std::string fileName, std::vector< cv::Point3d >& pts) {
	std::ofstream fs(fileName, std::ios_base::out | std::ios_base::trunc);
	fs.precision(std::numeric_limits< double >::max_digits10);
	for(unsigned int i = 0; i < pts.size(); i++) {
		fs << pts[i].x << ", " << pts[i].y << ", " << pts[i].z << "\n";
	}
	fs.close();
}

void Geometry::readPts3(std::string fileName, std::vector< cv::Point3d >& pts) {
	std::ifstream fs(fileName, std::ios_base::in);
	fs.precision(std::numeric_limits< double >::max_digits10);
	std::string line;
	while(std::getline(fs, line, '\n')) {
		const char* c = line.c_str();
		double x, y, z;
		std::sscanf(c, "%lf, %lf, %lf", &x, &y, &z);
		cv::Point3d p(x, y, z);
		pts.push_back(p);
	}
	fs.close();
}

void Geometry::genPts3Random(std::vector< cv::Point3d >& pts, int numPts, double minDist, double maxDist) {
	// Initialize random seed
	unsigned angSeed = std::chrono::system_clock::now().time_since_epoch().count();
	unsigned distSeed = std::chrono::system_clock::now().time_since_epoch().count();
	std::uniform_real_distribution<double> angDist(-PI, PI);
	std::mt19937_64 angRng(angSeed);
	std::uniform_real_distribution<double> distDist(minDist, maxDist);
	std::mt19937_64 distRng(distSeed);

	for(int i = 0; i < numPts; i++) {
		float phi = angDist(angRng);
		float theta = angDist(angRng);
		float dist = distDist(distRng);

		cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
		angs2Vec(phi, theta, &vec);
		vec *= dist;
		cv::Point3d pt(vec.at< float >(0, 0),
				       vec.at< float >(1, 0),
				       vec.at< float >(2, 0));
		pts.push_back(pt);
	}
}
