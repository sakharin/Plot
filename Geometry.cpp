#include "Geometry.hpp"

Geometry::Geometry() {
}

void Geometry::getRMatrixEulerAngles(float A, float B, float C, cv::Mat *R) {
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
    } else {
        *theta = std::acos(z / r);
        *phi = std::fmod(std::atan2(y, x) + TWOPI, TWOPI);
    }
}

void Geometry::vec2Angs(cv::Mat vec, float *theta, float *phi) {
    CV_Assert(vec.data != NULL);
    CV_Assert(vec.rows == 3 && vec.cols == 1);
    CV_Assert(vec.type() == CV_32F);

    float x, y, z;

    x = vec.at<float>(0, 0);
    y = vec.at<float>(1, 0);
    z = vec.at<float>(2, 0);
    vecElem2Angs(x, y, z, theta, phi);
}

void Geometry::vec2Angs(cv::Mat vecs, cv::Mat *thetas, cv::Mat *phis) {
    CV_Assert(vecs.data != NULL);
    CV_Assert(thetas->data != NULL);
    CV_Assert(phis->data != NULL);

    CV_Assert(vecs.rows == thetas->rows && vecs.cols == thetas->cols);
    CV_Assert(vecs.rows == phis->rows && vecs.cols == phis->cols);

    CV_Assert(vecs.type() == CV_32FC3);
    CV_Assert(thetas->type() == CV_32F);
    CV_Assert(phis->type() == CV_32F);

    int h = vecs.rows;
    int w = vecs.cols;

    if(vecs.isContinuous() && thetas->isContinuous() && phis->isContinuous()) {
        w *= h;
        h = 1;
    }

    int i, j;
    float *x, *y, *z, *t, *p;
    float r;
    for(i = 0; i < h; i++) {
        x = vecs.ptr<float>(i);
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
            x = x1 - *x2;
            y = y1 - *y2;
            z = z1 - *z2;
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

void Geometry::twoPts2Angs(cv::Mat P1, cv::Mat P2, float *theta, float *phi) {
    CV_Assert(P1.data != NULL);
    CV_Assert(P1.rows == 3 && P1.cols == 1);
    CV_Assert(P1.type() == CV_32F);

    CV_Assert(P2.data != NULL);
    CV_Assert(P2.rows == 3 && P2.cols == 1);
    CV_Assert(P2.type() == CV_32F);

    float x = P1.at<float>(0, 0) - P2.at<float>(0, 0);
    float y = P1.at<float>(1, 0) - P2.at<float>(1, 0);
    float z = P1.at<float>(2, 0) - P2.at<float>(2, 0);
    vecElem2Angs(x, y, z, theta, phi);
}

void Geometry::twoPts2Angs(cv::Mat P1, cv::Mat P2, cv::Mat *thetas, cv::Mat *phis) {
    CV_Assert(P1.data != NULL);
    CV_Assert(P1.rows == 3 && P1.cols == 1);
    CV_Assert(P1.type() == CV_32F);

    CV_Assert(P2.data != NULL);
    CV_Assert(thetas->data != NULL);
    CV_Assert(phis->data != NULL);

    CV_Assert(P2.rows == thetas->rows && P2.cols == thetas->cols);
    CV_Assert(P2.rows == phis->rows && P2.cols == phis->cols);

    CV_Assert(P2.type() == CV_32FC3);
    CV_Assert(thetas->type() == CV_32F);
    CV_Assert(phis->type() == CV_32F);

    int h = P2.rows;
    int w = P2.cols;

    if(P2.isContinuous() &&
       thetas->isContinuous() &&
       phis->isContinuous()) {
        w *= h;
        h = 1;
    }

    int i, j;
    float x1 = P1.at<float>(0, 0);
    float y1 = P1.at<float>(1, 0);
    float z1 = P1.at<float>(2, 0);
    float x, y, z;
    float *x2, *y2, *z2;
    float *t, *p;
    for(i = 0; i < h; i++) {
        x2 = P2.ptr<float>(i);
        y2 = x2 + 1;
        z2 = x2 + 2;

        t = thetas->ptr<float>(i);
        p = phis->ptr<float>(i);
        for(j = 0; j < w; j++) {
            x = x1 - *x2;
            y = y1 - *y2;
            z = z1 - *z2;

            vecElem2Angs(x, y, z, t++, p++);
            x2 += 3;
            y2 += 3;
            z2 += 3;
        }
    }
}
