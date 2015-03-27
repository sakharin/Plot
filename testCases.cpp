#include <iostream>
#include <cmath>

#include "Geometry.hpp"

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define PIOTWO 1.57079632679


void test_vec2Angs();
void test_vec2Angs_Vec();
void test_vec2Angs_Mat();

void test_twoPts2Vec();
void test_twoPts2Vec_PtPt();
void test_twoPts2Vec_PtMat();

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

void test_twoPts2Vec() {
    test_twoPts2Vec_PtPt();
    test_twoPts2Vec_PtMat();
}

void test_twoPts2Vec_PtPt() {
    Geometry geo = Geometry();
    float arr1[] = {1, 2, 3};
    float arr2[] = {7, 8, 9};
    cv::Mat P1 = cv::Mat(3, 1, CV_32F, arr1);
    cv::Mat P2 = cv::Mat(3, 1, CV_32F, arr2);
    cv::Mat vec = cv::Mat::zeros(3, 1, CV_32F);

    geo.twoPts2Vec(P1, P2, &vec);

    float x, y, z, norm;
    x = arr1[0] - arr2[0];
    y = arr1[1] - arr2[1];
    z = arr1[2] - arr2[2];
    norm = std::sqrt(x * x + y * y + z * z);
    x /= norm;
    y /= norm;
    z /= norm;

    assert(std::abs(vec.at<float>(0, 0) - x) < 0.000001);
    assert(std::abs(vec.at<float>(1, 0) - y) < 0.000001);
    assert(std::abs(vec.at<float>(2, 0) - y) < 0.000001);
}

void test_twoPts2Vec_PtMat() {
    Geometry geo = Geometry();
    int h = 100, w = 120;
    cv::Mat P1 = cv::Mat(3, 1, CV_32F);
    cv::Mat P2 = cv::Mat(h, w, CV_32F);
    cv::Mat vec = cv::Mat::zeros(h, w, CV_32FC3);
    cv::randu(P1, cv::Scalar(-100), cv::Scalar(10));
    cv::randu(P2, cv::Scalar(-100), cv::Scalar(10));

    float x1 = P1.at<float>(0, 0);
    float y1 = P1.at<float>(1, 0);
    float z1 = P1.at<float>(2, 0);
    float x, y, z, norm;
    float *x2, *y2, *z2;
    float *x3, *y3, *z3;
    for(int i =0; i < h; i++) {
        x2 = P2.ptr<float>(i);
        y2 = x2 + 1;
        z2 = x2 + 2;

        x3 = vec.ptr<float>(i);
        y3 = x3 + 1;
        z3 = x3 + 2;
        for(int j =0; j < h; j++) {
            x = x1 - *x2;
            y = y1 - *y2;
            z = z1 - *z2;
            norm = std::sqrt(x * x + y *y + z * z);

            assert(std::abs(*x3 - x) < 0.000001);
            assert(std::abs(*y3 - y) < 0.000001);
            assert(std::abs(*y3 - y) < 0.000001);
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
    cv::Mat P1 = cv::Mat(3, 1, CV_32F);
    cv::Mat P2 = cv::Mat(h, w, CV_32FC3);
    cv::Mat thetas = cv::Mat::zeros(h, w, CV_32F);
    cv::Mat phis = cv::Mat::zeros(h, w, CV_32F);
    cv::randu(P1, cv::Scalar(-100), cv::Scalar(10));
    cv::randu(P2, cv::Scalar(-100), cv::Scalar(10));


    float x1 = P1.at<float>(0, 0);
    float y1 = P1.at<float>(1, 0);
    float z1 = P1.at<float>(2, 0);

    geo.twoPts2Angs(P1, P2, &thetas, &phis);

    float x, y, z, r;
    float *p2;
    for(int i = 0; i < h; i++) {
        p2 = P2.ptr<float>(i);
        for(int j = 0; j < w; j++) {
            x = x1 - *p2++;
            y = y1 - *p2++;
            z = z1 - *p2++;

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
