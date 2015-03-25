#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cv.h>
#include <highgui.h>
#include <math.h>

#include <iostream>

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define PIOTWO 1.57079632679

class Geometry {
    protected:
        int a;
        void vecElem2Angs(float, float, float, float*, float*);
    public:
        Geometry();
        void vec2Angs(cv::Mat*, float *, float *);
        void vec2Angs(cv::Mat*, cv::Mat*, cv::Mat*);
};
#endif
