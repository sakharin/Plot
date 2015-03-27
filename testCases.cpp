#include <iostream>
#include <cmath>

#include "Geometry.hpp"

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define PIOTWO 1.57079632679


void test_vec2Angs();
void test_vec2Angs_Vec();
void test_vec2Angs_Mat();

void test_twoPts2Angs();
void test_twoPts2Angs_PtPt();
void test_twoPts2Angs_PtMat();


void test_vec2Angs() {
    test_vec2Angs_Vec();
    test_vec2Angs_Mat();
}

void test_vec2Angs_Vec() {
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

void test_vec2Angs_Mat() {
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

void test_twoPts2Angs() {
    test_twoPts2Angs_PtPt();
    test_twoPts2Angs_PtMat();
}

void test_twoPts2Angs_PtPt() {
    Geometry geo = Geometry();
    float arr1[] = {1, 2, 3};
    float arr2[] = {7, 8, 9};
    cv::Mat P1 = cv::Mat(3, 1, CV_32F, arr1);
    cv::Mat P2 = cv::Mat(3, 1, CV_32F, arr2);
    float theta, phi;

    geo.twoPts2Angs(P1, P2, &theta, &phi);

    float x, y, z;
    x = arr1[0] - arr2[0];
    y = arr1[1] - arr2[1];
    z = arr1[2] - arr2[2];

    assert(std::abs(phi - std::fmod(std::atan2(y, x) + TWOPI, TWOPI)) < 0.000001);
}

void test_twoPts2Angs_PtMat() {
    Geometry geo = Geometry();
    int h = 6, w = 10;
    cv::Mat P1 = cv::Mat::zeros(h, w, CV_32FC3);
    cv::Mat P2 = cv::Mat::zeros(h, w, CV_32FC3);
    cv::Mat thetas = cv::Mat::zeros(h, w, CV_32F);
    cv::Mat phis = cv::Mat::zeros(h, w, CV_32F);

    float *p1, *p2;
    for(int i = 0; i < h; i++) {
        p1 = P1.ptr<float>(i);
        p2 = P2.ptr<float>(i);
        for(int j = 0; j < w; j++) {
            (*p1++) = i * h + j;
            (*p1++) = j * w + i;
            (*p1++) = i + j;

            (*p2++) = i * w + j;
            (*p2++) = j * h + i;
            (*p2++) = i + j * 2;
        }
    }
    geo.twoPts2Angs(P1, P2, &thetas, &phis);

    P1 = P1 - P2;
    float x, y, z, r;
    for(int i = 0; i < h; i++) {
        p1 = P1.ptr<float>(i);
        for(int j = 0; j < w; j++) {
            x = *p1++;
            y = *p1++;
            z = *p1++;

            r = std::sqrt(x * x + y * y + z * z);
            if(r == 0) {
                assert(std::abs(thetas.at<float>(i, j)) < 0.000001);
                assert(std::abs(phis.at<float>(i, j)) < 0.000001);
            } else {
                assert(std::abs(thetas.at<float>(i, j) - std::acos(z / r)) < 0.000001);
                assert(std::abs(phis.at<float>(i, j) - std::fmod(std::atan2(y, x) + TWOPI, TWOPI)) < 0.000001);
            }

        }
    }
}

int main (int argc, char *argv[]) {
    test_vec2Angs();
    test_twoPts2Angs();
    return 0;
}
