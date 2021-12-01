#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class algorithms {
public:

  typedef struct {
    int r;                  // radius of the circle (r-th accumulator-vector-matrix)
    int x;                  // x-idx accumulator matrix
    int y;                  // y-idx accumulator matrix
    int accumulator_value;  // accumulator value
  } CvLocalMaximum;

  typedef struct {
    double value;
    double diameter;
  } coin;

  static void drawHoughLines(Mat &image, vector<Vec2f> lines);

  static void cannyOwn(const Mat &image, Mat &end_result, uchar threshold_min, uchar threshold_max,
                       Mat &grad_x, Mat &grad_y, Mat &grad, Mat &gradient_directions, Mat &non_maxima_suppression);

  static void Erosion(int erosion_type, int erosion_size, InputArray src, OutputArray dst);

  static void Dilation(int dilation_type, int dilation_size, InputArray src, OutputArray dst);

  static void
  HoughLinesOwn(Mat img, vector<Vec2f> &lines, float rho, float theta, int threshold, int linesMax, Mat &accum,
                vector<CvLocalMaximum> &local_maximums);

  static void
  HoughCirclesOwn(const Mat &img, vector<Vec3f> &circles, float rho, float rad, int rad_min, int rad_max, int threshold, int circlesMax,
                  vector<Mat> &accum, vector<CvLocalMaximum> &local_maximums);

  static void drawHoughCircles(Mat &result, const vector<Vec3f> &circles);

  static double classifyCircles(const vector<Vec3f> &circles, const vector<Vec2f> &lines,
                                const vector<algorithms::coin> coin_properties);

  static void writeHoughAccum(const vector<Mat> &circleAccum, const string &fname);
};


#endif //CGCV_ALGORITHMS_H
