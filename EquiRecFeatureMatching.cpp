#include "EquiRecFeatureMatching.hpp"

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

EquiRecFeatureMatching::EquiRecFeatureMatching(cv::Mat inImg1, cv::Mat inImg2) {
	geo = Geometry();
	H = inImg1.size().height;
	W = inImg1.size().width;
	float phi, theta;
	int N = 6;
	float beta = PI / (2 * N);

	// Get unit sphere vectors and mask
	cv::Mat vecs = cv::Mat::zeros(H, W, CV_32FC3);
	cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
	cv::Mat mark = cv::Mat::zeros(H, W, CV_8U);
	for (int i = 0; i < H; i++) {
		theta = i * PI / H;
		for (int j = 0; j < W; j++) {
			phi = j * TWOPI / W;

			geo.angs2Vec(&vec, phi, theta);
			vecs.at<cv::Vec3f>(i, j) = vec;

			float ang = std::atan(std::tan(theta - PIOTWO) / std::cos(phi + PI));
			mark.at<unsigned char>(i, j) = 255 * ((-beta < ang) and (ang <= beta));
		}
	}

	// Variables for rotating each image
	cv::Mat rotatedVecs = cv::Mat::zeros(H, W, CV_32FC3);
	cv::Mat m = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat phis = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat thetas = cv::Mat::zeros(H, W, CV_32F);
    cv::Mat gridx = cv::Mat::zeros(H, W, CV_32F);
    cv::Mat gridy = cv::Mat::zeros(H, W, CV_32F);
	cv::Mat rotatedImg1 = cv::Mat::zeros(H, W, CV_8U);
	cv::Mat rotatedImg2 = cv::Mat::zeros(H, W, CV_8U);
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
		std::vector< cv::KeyPoint > kps1;
		std::vector< cv::KeyPoint > kps2;
		cv::Mat deses1;
		cv::Mat deses2;

		cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
		cv::remap(inImg1, rotatedImg1, gridx, gridy,
				  CV_INTER_LINEAR, cv::BORDER_REFLECT_101);
		cv::remap(inImg2, rotatedImg2, gridx, gridy,
				  CV_INTER_LINEAR, cv::BORDER_REFLECT_101);
		sift->detect(rotatedImg1, kps1, mark);
		sift->detect(rotatedImg2, kps2, mark);
		sift->compute(rotatedImg1, kps1, deses1);
		sift->compute(rotatedImg2, kps2, deses2);

		cv::FlannBasedMatcher matcher;
		std::vector< std::vector< cv::DMatch > > matches;
		matcher.knnMatch(deses1, deses2, matches, 2);

		// Copy keypoints
		int previousKeyPointsIndex1 = keyPoints1.size();
		int previousKeyPointsIndex2 = keyPoints2.size();
		geo.getRMatrixEulerAngles(&m, 0, -k * 2 * beta, 0);
		for (int i = 0; i < kps1.size(); i++) {
			// Rotate keyPoint1
			cv::KeyPoint kp;
			kp = kps1[i];
			point2Vec(&vec, kp.pt);
			vec = m * vec;
			vec2Point(&kp.pt, vec);
			keyPoints1.push_back(kp);
		}
		for (int i = 0; i < kps2.size(); i++) {
			// Rotate keyPoint2
			cv::KeyPoint kp;
			kp = kps2[i];
			point2Vec(&vec, kp.pt);
			vec = m * vec;
			vec2Point(&kp.pt, vec);
			keyPoints2.push_back(kp);
		}

        // Ratio test as per Lowe's paper
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < 0.75 * matches[i][1].distance) {
				cv::DMatch m = matches[i][0];
				m.queryIdx += previousKeyPointsIndex1;
				m.trainIdx += previousKeyPointsIndex2;
				goodMatches.push_back(m);
			}
		}
	}
}
