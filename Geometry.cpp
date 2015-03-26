#include "Geometry.hpp"

Geometry::Geometry() {
}

void Geometry::getRMatrixEulerAngles(cv::Mat *R, float A=0, float B=0, float C=0) {
    CV_Assert(R->data != NULL);
    CV_Assert(R->rows == 3 && R->cols == 3);
    CV_Assert(R->type() == CV_32F);

    //Graphic Gems IV, Paul S. Heckbert, 1994
    float sA = sin(A);
    float sB = sin(B);
    float sC = sin(C);

    float cA = cos(A);
    float cB = cos(B);
    float cC = cos(C);

    float rC[] = {cC, -sC, 0.,
                  sC, cC, 0.,
                  0., 0., 1.};
    float rB[] = {cB, 0., sB,
                  0., 1., 0.,
                  -sB, 0., cB};
    float rA[] = {1., 0., 0.,
                  0., cA, -sA,
                  0., sA, cA};

    cv::Mat mrC(3, 3, CV_32FC1, rC);
    cv::Mat mrB(3, 3, CV_32FC1, rB);
    cv::Mat mrA(3, 3, CV_32FC1, rA);

    cv::Mat tmp = mrA * mrB * mrC;
    tmp.copyTo(*R);
}

void Geometry::vecElem2Angs(float x, float y, float z, float* theta, float* phi) {
    float r;
    r = std::sqrt(x * x + y * y + z * z);
    if(r == 0) {
        *theta = 0;
        *phi = 0;
    } else if(x == 0) {
        *theta = std::acos(z / r);
        if(y == 1) {
            *phi = 0;
        } else if(y > 0) {
            *phi = PIOTWO;
        } else {
            *phi = 3 * PIOTWO;
        }
    } else {
        *theta = std::acos(z / r);
        *phi = std::atan2(y, x);
    }
}

void Geometry::vec2Angs(cv::Mat *vec, float *theta, float *phi) {
    CV_Assert(vec->data != NULL);
    CV_Assert(vec->rows == 3 && vec->cols == 1);
    CV_Assert(vec->type() == CV_32F);

    float x, y, z;

    x = vec->at<float>(0, 0);
    y = vec->at<float>(1, 0);
    z = vec->at<float>(2, 0);
    vecElem2Angs(x, y, z, theta, phi);
}

void Geometry::vec2Angs(cv::Mat *vecs, cv::Mat *thetas, cv::Mat *phis) {
    CV_Assert(vecs->data != NULL);
    CV_Assert(thetas->data != NULL);
    CV_Assert(phis->data != NULL);

    CV_Assert(vecs->rows == thetas->rows && vecs->cols == thetas->cols);
    CV_Assert(vecs->rows == phis->rows && vecs->cols == phis->cols);

    CV_Assert(vecs->type() == CV_32F);
    CV_Assert(thetas->type() == CV_32F);
    CV_Assert(phis->type() == CV_32F);

    int h = vecs->rows;
    int w = vecs->cols;

    if(vecs->isContinuous() && thetas->isContinuous() && phis->isContinuous()) {
        w *= h;
        h = 1;
    }

    int i, j;
    float *x, *y, *z, *t, *p;
    float r;
    for(i = 0; i < h; i++) {
        x = vecs->ptr<float>(i);
        y = x + 1;
        z = x + 2;

        t = thetas->ptr<float>(i);
        p = phis->ptr<float>(i);
        for(j = 0; j < w; j++) {
            vecElem2Angs(*x, *y, *z, t, p);
            x += 3;
            y += 3;
            z += 3;
            t += 1;
            p += 1;
        }
    }
}

void Geometry::twoPts2Vec(cv::Mat P1, cv::Mat P2, cv::Mat *P) {
    CV_Assert(P1.data != NULL);
    CV_Assert(P1.rows == 3 && P1.cols == 1);
    CV_Assert(P1.type() == CV_32F);

    CV_Assert(P2.data != NULL);
    CV_Assert(P2.rows == 3 && P2.cols == 1);
    CV_Assert(P2.type() == CV_32F);

    CV_Assert(P->data != NULL);
    CV_Assert(P->rows == 3 && P->cols == 1);
    CV_Assert(P->type() == CV_32F);

    *P = P2 - P1;
    float x = P->at<float>(0, 0);
    float y = P->at<float>(1, 0);
    float z = P->at<float>(2, 0);
    float norm = x * x + y * y + z * z;
    *P /= norm;
}
