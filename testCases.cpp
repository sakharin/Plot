#include <iostream>
#include <cmath>

#include "Geometry.hpp"

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define PIOTWO 1.57079632679

void test_vec2Angs();
void test_vec2Angs_Scalar();

void test_vec2Angs() {
    test_vec2Angs_Scalar();
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

int main (int argc, char *argv[]) {
    test_vec2Angs();
    return 0;
}
