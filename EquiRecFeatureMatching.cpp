#include "EquiRecFeatureMatching.hpp"

EquiRecFeatureMatching::EquiRecFeatureMatching() {
	previousH = 0;
	previousW = 0;
	geo = Geometry();
}

void EquiRecFeatureMatching::point2Vec(cv::Mat *vec, cv::Point2f pt) {
	float phi, theta;
	geo.u2Phi(&phi, pt.x, W);
	geo.v2Theta(&theta, pt.y, H);
	geo.angs2Vec(vec, phi, theta);
}

void EquiRecFeatureMatching::vec2Point(cv::Point2f *pt, cv::Mat vec) {
	float phi, theta;
	float x, y;
	geo.vec2Angs(&phi, &theta, vec);
	geo.phi2u(&x, phi, W);
	geo.theta2v(&y, theta, H);
	pt->x = std::fmod(x, float(W));
	pt->y = std::fmod(y, float(H));
}

void EquiRecFeatureMatching::detectAndCompute(cv::Mat inImg, std::vector< cv::KeyPoint > *keyPoints, cv::Mat *descriptors) {
	int N = 6;
	float beta = PI / (2 * N);

	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	float phi, theta;

	// Get unit sphere vectors and mask
	H = inImg.size().height;
	W = inImg.size().width;
	assert(H != 0 && W != 0);
	if (H != previousH || W != previousW) {
		vecs = cv::Mat::zeros(H, W, CV_32FC3);
		mask = cv::Mat::zeros(H, W, CV_8U);
		for (int i = 0; i < H; i++) {
			theta = i * PI / H;
			for (int j = 0; j < W; j++) {
				phi = -j * TWOPI / W;

				geo.angs2Vec(&vec, phi, theta);
				vecs.at<cv::Vec3f>(i, j) = vec;

				float ang = std::atan(std::tan(theta - PIOTWO) / std::cos(phi + PI));
				mask.at<unsigned char>(i, j) = 255 * ((-beta < ang) and (ang <= beta));
			}
		}
	}

	// Variables for rotating each image
	cv::Mat rotatedVecs = cv::Mat::zeros(H, W, CV_32FC3);
	cv::Mat m = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat phis = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat thetas = cv::Mat::zeros(H, W, CV_32F);
    cv::Mat gridx = cv::Mat::zeros(H, W, CV_32F);
    cv::Mat gridy = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat rotatedImg = cv::Mat::zeros(H, W, CV_8U);
	float u, v;


	// For all rotation angle
	for (int k = 0; k < N; k++) {
		// Rotate the unit sphere
		geo.getRMatrixEulerAngles(&m, 0, k * 2 * beta, 0);
		for(int i = 0; i < H; i++) {
			for(int j = 0; j < W; j++) {
				vec.at<float>(0, 0) = vecs.at<cv::Vec3f>(i, j)[0];
				vec.at<float>(1, 0) = vecs.at<cv::Vec3f>(i, j)[1];
				vec.at<float>(2, 0) = vecs.at<cv::Vec3f>(i, j)[2];
				vec = m * vec;
				geo.vec2Angs(&phi, &theta, vec);
				geo.phi2u(&u, phi, W);
				geo.theta2v(&v, theta, H);
				gridx.at<float>(i, j) = u;
				gridy.at<float>(i, j) = v;
			}
		}

		// Get features
		std::vector< cv::KeyPoint > kps;
		cv::Mat deses;

		cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
		cv::remap(inImg, rotatedImg, gridx, gridy,
				  CV_INTER_LINEAR, cv::BORDER_REFLECT_101);
		sift->detect(rotatedImg, kps, mask);
		sift->compute(rotatedImg, kps, deses);

		// Copy keypoints
		int previousKeyPointsIndex = keyPoints->size();
		geo.getRMatrixEulerAngles(&m, 0, k * 2 * beta, 0);
		for (int i = 0; i < kps.size(); i++) {
			// Rotate keyPoint
			cv::KeyPoint kp;
			kp = kps[i];
			point2Vec(&vec, kp.pt);
			vec = m * vec;
			vec2Point(&kp.pt, vec);
			keyPoints->push_back(kp);
		}

		// Copy descriptors
		descriptors->push_back(deses);
	}
}
