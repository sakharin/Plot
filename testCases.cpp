#include <iostream>
#include <cmath>

#include "Geometry.hpp"

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define PIOTWO 1.57079632679


void test_vec2Angs();
void test_vec2Angs_Scalar();
void test_vec2Angs_CvMat();

void test_vec2Angs() {
    test_vec2Angs_Scalar();
    test_vec2Angs_CvMat();
}

void test_vec2Angs_Scalar() {
    Geometry geo = Geometry();
    cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);
    float theta, phi;
    float x, y, z;

    vec.at<float>(0, 0) = 0;
    vec.at<float>(1, 0) = 0;
    vec.at<float>(2, 0) = 0;
    geo.vec2Angs(&vec, &theta, &phi);
    assert(theta == 0 && phi == 0);

    for(int i = 0; i < 360; i++) {
        x = std::cos(i / 360. * TWOPI);
        y = std::sin(i / 360. * TWOPI);
        z = 0;
        vec.at<float>(0, 0) = x * (i + 1);
        vec.at<float>(1, 0) = y * (i + 1);
        vec.at<float>(2, 0) = z * (i + 1);

        geo.vec2Angs(&vec, &theta, &phi);
        assert(std::abs(phi - (i / 360. * TWOPI)) < 0.000001);
        assert(std::abs(theta - PIOTWO) < 0.000001);
    }
}

void test_vec2Angs_CvMat() {
    Geometry geo = Geometry();
    int h = 6, w = 10;
    cv::Mat vecs = cv::Mat::zeros(h, w, CV_32FC3);
    cv::Mat thetas = cv::Mat::zeros(h, w, CV_32F);
    cv::Mat phis = cv::Mat::zeros(h, w, CV_32F);

    float *p;
    float x, y, z;
    for(int i = 0; i < h; i++) {
        p = vecs.ptr<float>(i);
        for(int j = 0; j < w; j++) {
            x = 1;
            y = std::sin((w / 2. - j) / 360. * TWOPI);
            z = h / 2. - i;
            (*p++) = x;
            (*p++) = y;
            (*p++) = z;
        }
    }
    geo.vec2Angs(&vecs, &thetas, &phis);
    float r;
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            x = 1;
            y = std::sin((w / 2. - j) / 360. * TWOPI);
            z = h / 2. - i;

            r = std::sqrt(x * x + y * y + z * z);
            assert(std::abs(thetas.at<float>(i, j) - std::acos(z / r)) < 0.000001);

            assert(std::abs(phis.at<float>(i, j) - std::fmod(std::atan2(y, x) + TWOPI, TWOPI)) < 0.000001);
        }
    }
}

int main (int argc, char *argv[]) {
    test_vec2Angs();
    return 0;
}
