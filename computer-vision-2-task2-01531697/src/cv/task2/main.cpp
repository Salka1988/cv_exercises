#include <rapidjson/document.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/random.h>
#include <sstream>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

static const int numOctaves = 6;
static const int numLayers = 4;

#define FULL_VERSION 1

// TODO: change this if you want to implement the Bonus-Task
#define BONUS 1

#define START_TIMER(variable)
#define STOP_TIMER(variable)

void saveOutput(const Mat &output, const string &name, const string &out_path, bool bonus);
void executeTestcase(rapidjson::Value&, bool bonus);

//==============================================================================
// Fast(const Mat& img, std::vector<KeyPoint>& features, 
//      const vector<Point> &circle, const Mat& R, int N, int thresh, 
//      float harrisThreshold)
//------------------------------------------------------------------------------
// Find feature points using the FAST algorithm
//
// TODO (BONUS):
// - calculate a lower and upper threshold
// - for every pixel check if the harris corner measure is higher than a given
//   threshold
// - if higher perform the FAST algorithm
// - check if N consecutive points are higher or lower than the threshold
// - at the same time calculate the sum of absolute differences (SAD) between the
//   center and the current pixel on the circle
// - if N consecutive pixels on the circle are higher/lower than the threshold 
//   mark this point as feature point
// - perform a non-maxima suppression: for every feature point that has an 
//   Euclidean distance less equal than 9 check if
//   its SAD is higher or lower than the current feature point, only keep the point
//   with the highest SAD
// - return the found feature points in features
//
//
// Parameters:
// const Mat& img: the input image
// std::vector<KeyPoint>& features: the found feature points
// vector<Point> circle: points on a circle with radius 9
// const Mat& R: harris corner measure
// int N: number of consecutive points that should be higher/lower
// int thresh: value that should be used to calculate the upper and lower
//             threshold 
// float harrisThreshold: threshold for the harris corner measure
//==============================================================================
void Fast(const Mat& img, std::vector<KeyPoint>& features, 
          const vector<Point> &circle, const Mat& R, int N, int thresh, 
          float harrisThreshold)
{
}



// struct with weak learners' parameters
struct WeakLearner{
  float thresh;
  int orient;
  int x_min;
  int x_max;
  int y_min;
  int y_max;
  float alpha;
  float beta;
  
  WeakLearner() :
          thresh (0.0), orient(0), x_min(0), x_max(0), y_min(0), y_max(0), alpha(0.0), beta(0.0)
  {}
};
// gradient assignment type
enum Assign {
  ASSIGN_HARD=0,
  ASSIGN_BILINEAR=1,
  ASSIGN_SOFT=2,
  ASSIGN_HARD_MAGN=3,
  ASSIGN_SOFT_MAGN=4
};
struct BinBoostProperties{
  string wl_bin_path;
  int nWLs;
  int nDim;
  WeakLearner** pWLsArray;
  int orientQuant;
  int patchSize;
  Assign gradAssignType;
  uchar binLookUp[8];
};



//==============================================================================
//void computeOrientation(vector<KeyPoint>& features,
//                        const BinBoostProperties& properties,
//                        const int nAngleBins,
//                        const vector<vector<Mat>>& directions,
//                        const vector<vector<Mat>>& magnitudes)
//------------------------------------------------------------------------------
// Computes the orientation of the keypoints (Features)
//
// TODO:
// - Loop through each keypoint and compute its octave, layer and x,y position (this is done for you)
// - Around a local neighborhood of (x,y) +/- patchSize / 2.f fetch the orientation (and tranform it to degrees)
//   and magnitude. Don't go out of bounds of the corresponding directions and magnitudes image!
// - Quantize the orientation, which is from [0, 360.f] into an orientation histogram of nAngleBins entries.
//   Be carefull to handle angles < 0 and >= 360 by adding 360 degrees or subtracting 360 degrees, so that they
//   fall into [0, 360[ degrees.
// - Add the magnitude on that pixel weighted by e^(-(dx^2+dy^2)/(2*sigma*sigma)) to the histogram at the
//   corresponding bin. dx, dy denotes the horizontal and vertical distance to the keypoint center (x,y).
// - After all pixels are in the histogram, find the maximum bin of the histogram and set the angle
//   of the keypoint to (maxBin + 0.5f) * 360.f / nAngleBins.
//
// Parameters:
// vector<KeyPoint>& features: the vector containing the features for which the orientation should be calculated
// const BinBoostProperties& properties: The properties containing the patch-size.
// const int nAngleBins: The number of bins to sort the gradient-points of a patch.
// const vector<vector<Mat>>& directions: Mats containing the direction of every point for every octave and layer.
//                                        directions[octave][layer] are the orientations of the corresponding 
//                                        octave and layer
// const vector<vector<Mat>>& magnitudes: Mats containing the magnitude of every point for every octave and layer.
//                                        magnitudes[octave][layer] are the magnitudes of the corresponding
//                                        octave and layer.
//==============================================================================
void computeOrientation(vector<KeyPoint>& features,
                        const BinBoostProperties& properties,
                        const int nAngleBins,
                        const vector<vector<Mat>> & directions,
                        const vector<vector<Mat>> & magnitudes)
{
    const int patchSize = 15; // patch size around current keypoint
    const float sigma = 1.8f; // used for weighting the magnitudes.
    const float radToDeg = 180.f / M_PI; // constant for conversion the orientation to degree
    std::vector<float> histogramm(nAngleBins); // our histogram


    for(KeyPoint& kp : features) {
        std::fill(histogramm.begin(), histogramm.end(), 0.0f);
        // you'll need octave and layer to index the directions and magnitudes
        // vector.
        int idx = kp.octave;
        int octave = idx / (numLayers+1);   // octave of the current keypoint
        int layerIdx = idx % (numLayers+1); // layer of the current keypoint

        // x and y position of the current keypoint.
        const int x = float(kp.pt.x) / (1<<octave);
        const int y = float(kp.pt.y) / (1<<octave);

        int offset = patchSize / 2;

        for (int y_start = y - offset; y_start < y + offset; y_start++)
        {
            if (y_start < 0 || y_start >= directions.at(octave).at(layerIdx).rows)
            {
                continue;
            }

            for(int x_start = x - offset; x_start < x + offset; x_start++)
            {
                if(x_start < 0 || x_start >= directions.at(octave).at(layerIdx).cols)
                {
                    continue;
                }

                float orientation = directions.at(octave).at(layerIdx).at<float>(y_start ,x_start) * radToDeg;

                float orientation_adjusted = orientation;
                if(orientation < 0)
                {
                    orientation_adjusted = orientation + 360;
                }
                else if(orientation >= 360)
                {
                    orientation_adjusted = orientation - 360;
                }

                int bin = cvRound(orientation_adjusted * (float(nAngleBins) / 360.f));

                int bin_adjusted = bin;
                if (bin < 0)
                {
                    bin_adjusted = bin + nAngleBins;
                }
                else if(bin >= nAngleBins)
                {
                    bin_adjusted = bin - nAngleBins;
                }

                float scale = exp(-(((y_start - y) * (y_start - y) + (x_start - x) * (x_start - x)) / (2.0f * sigma * sigma)));

                histogramm.at(bin_adjusted) = histogramm.at(bin_adjusted) + magnitudes.at(octave).at(layerIdx).at<float>(y_start ,x_start) * scale;
            }
        }

        int maximum_bin = std::distance(histogramm.begin(), std::max_element(histogramm.begin(), histogramm.end()));

        kp.angle = ((float) maximum_bin + 0.5f) * 360.0f / (float) nAngleBins;
    }
}



//==============================================================================
//void computeGradientOrientationsAndMagnitudes(const Mat& img,
//                      Mat& directions,
//                      Mat& magnitudes)
//------------------------------------------------------------------------------
// Calculates the gradient magnitude and orientations of the input image
//
// TODO:
// - First the derivates in x an y-direction of the image should be calculated with help of the cv::Sobel-Function.
//   With this values you can calculate the magnitude and the angle. This two values should be stored on the
//   corresponding point in the Mats directions and magnitudes.
//
// Parameters:
// const Mat& img: the grayscale input image.
// Mat& directions: Mat where the direction of the gradients has to be stored
// Mat& magnitudes: Mat where the magnitude of the gradients has to be stored
//==============================================================================
void computeGradientOrientationsAndMagnitudes(
    const cv::Mat &img,
    cv::Mat &directions,
    cv::Mat &magnitudes)
{
    Mat derivx, derivy;

    directions = Mat::zeros(img.size(), CV_32FC1);
    magnitudes = Mat::zeros(img.size(), CV_32FC1);

    Sobel(img, derivy, CV_32F, 0, 1, 3);
    Sobel(img, derivx, CV_32F, 1, 0, 3);


    for (int current_row = 0; current_row < img.rows; current_row++)
    {
        for (int current_column = 0; current_column < img.cols; current_column++)
        {
            float current_deriv_x = derivx.at<float>(current_row, current_column);
            float current_deriv_y = derivy.at<float>(current_row, current_column);

            float deriv_x_squared = current_deriv_x * current_deriv_x;
            float deriv_y_squared = current_deriv_y * current_deriv_y;
            magnitudes.at<float>(current_row, current_column) = sqrt(deriv_x_squared + deriv_y_squared);

            float angle = atan2(current_deriv_y, current_deriv_x);
            directions.at<float>(current_row, current_column) = (angle < 0) ? angle + 2 * M_PI : angle;
        }
    }
}



//==============================================================================
//void computeGradients(const Mat& img,
//                      const BinBoostProperties& properties,
//                      vector<Mat>& gradients,
//                      Mat& directions,
//                      Mat& magnitudes)
//------------------------------------------------------------------------------
// Calculates the gradient of the input image in each of the directions k*((2*pi)/orientQuant) for 0 < k < orientQuant
//
// TODO:
// - First, compute the gradient magnitude and orientations with the function computeGradientOrientationsAndMagnitudes(...). 
//   (This is already done for you).
// - Iterate through all pixels:
//   - If the magnitude is smaller or equal then the given threshold the points in the gradients-Mat should be set to 0.
//   - If it's greater than threshold we compute the quantized gradient bin assignments (see assignment document for details):
//        - the current bin index (integer) is defined as theta / binSize, where binSize is a floating 
//          point which is 2.f * M_PI / orientQuant
//        - we loop through all neighbor bins from index - 2 to index + 2.
//        - for all neighbors bin, we compute the angle bc of the center of the bin
//        - we compute the cosine of the angle between bc and theta, which is our weight w
//        - we scale the weight by 255 and round it and store it into the bin-th gradient image.
//
// Parameters:
// const Mat& img: the grayscale input image.
// const BinBoostProperties& properties:
//                             .orientQuant: number of orientations in which the gradient of the image has to be calculated
// vector<Mat>& gradients: a vector of Mats, each entry shall contain the gradient of the input image in one direction.
// Mat& directions: Mat where the direction of the gradients has to be stored
// Mat& magnitudes: Mat where the magnitude of the gradients has to be stored
//==============================================================================
void computeGradients(const Mat& img,
                      const BinBoostProperties& properties,
                      vector<Mat>& gradients,
                      Mat& directions,
                      Mat& magnitudes)
{

        const int orientQuant = properties.orientQuant;
        const float magnitudeThreshold = 20;

        for (int i=0; i<orientQuant; ++i)
            gradients.push_back(Mat::zeros(img.size(), CV_8UC1));

        directions = Mat::zeros(img.size(), CV_32FC1);
        magnitudes = Mat::zeros(img.size(), CV_32FC1);

        computeGradientOrientationsAndMagnitudes(img, directions, magnitudes);

        const float bin_size = (2.f * M_PI) / orientQuant; // according to newsgroup

        for (int current_column = 0; current_column < magnitudes.cols; current_column++)
        {
            for (int current_row = 0; current_row < magnitudes.rows; current_row++)
            {
                if (magnitudes.at<float>(current_row, current_column) > magnitudeThreshold)
                {
                    int index_value = cvFloor(directions.at<float>(current_row, current_column) / bin_size);
                    int index = index_value == orientQuant ? 0 : index_value;

                    for (int neighbor_bin = index - 2; neighbor_bin <= index + 2; neighbor_bin++)
                    {
                        int bin_hat = (neighbor_bin + orientQuant) % orientQuant;
                        float bin_center = (bin_hat + 0.5f) * bin_size;
                        float weight = max(0.0f, cos(directions.at<float>(current_row, current_column) - bin_center));
                        gradients.at(bin_hat).at<uint8_t >(current_row, current_column) = cvRound(255 * weight);
                    }
                }
            }
        }

}


//==============================================================================
// void computeIntegrals(const vector<Mat>& gradients,
//                       const int orientQuant,
//                       vector<Mat>& integrals)
//------------------------------------------------------------------------------
// Computes the integral images of the given gradient images and additionally the sum of these integral images.
//
// TODO:
// - initialize the integral images. Remark that each integral image must have one row and one column
//   more than the gradients Mats. (First row and column are filled with zero)
//   We will calculate orientQuant + 1 different integral images.
// - calculate the integral image of each gradient image in the gradients vector
//   and write the results in the according integral image in the integrals vector.
//   Do _NOT_ use the cv::integralImage for this method.
// -
// - The final integral image (numer orientQuant + 1) shall be the sum of all prior
//   calculated integral images. This will be necessary to calculate a _relative_ WL response
//   later on.
//
//
// Parameters:
// const vector<Mat>& gradients: the gradients of which the integrals shall be calculated. The images are of type CV_8UC1.
// const int orientQuant: The number of directions, in which the gradients where taken
// vector<Mat>& integrals: The vector that shall be filled with the integrals of the gradients. The images are of type CV_32SC1.
//==============================================================================
void computeIntegrals(const vector<Mat>& gradients,
                      const int orientQuant,
                      vector<Mat>& integrals)
{
    {
        // Initialize Integral Images
        int rows = gradients[0].rows;
        int cols = gradients[0].cols;

        for (int i=0; i<orientQuant+1; ++i)
            integrals.push_back(Mat::zeros(rows+1, cols+1, CV_32SC1));

        for (int current_gradient = 0; current_gradient < gradients.size(); current_gradient++)
        {
            for(int current_column = 1; current_column < 1 + gradients.at(current_gradient).cols; current_column++)
            {
                int sum = 0;
                for(int current_row = 1; current_row < 1 + gradients.at(current_gradient).rows; current_row++)
                {
                    sum = sum + gradients.at(current_gradient).at<uchar>(-1 + current_row, -1 + current_column);
                    integrals.at(current_gradient).at<int>(current_row, current_column) = integrals
                                                                                                  .at(current_gradient)
                                                                                                  .at<int>(current_row, -1 + current_column) + sum;
                }
            }
        }
        for(int current_integral = 0; current_integral < orientQuant; current_integral++)
        {
            integrals.at(orientQuant) += integrals.at(current_integral);
        }
    }
}

//==============================================================================
//void rectifyPatch(const Mat& img,
//                  const KeyPoint& kp,
//                  const int& patchSize,
//                  Mat& patch)
//------------------------------------------------------------------------------
// Cuts out a rectangular patch that surround the keypoint
//
// TODO:
// - Every Keypoint has an orientation (calculated in computeOrientation()) and the rectangle, that is cut out of the
//   image should be aligned by this angle.
// - To cut out a rotated rectangle you can use the cv::warpAffine-function. This function takes a 2x3 transformation
//   matrix that is calculated using the keypoints angle and size.
// - Our transformation-matrix consists of a rotation and a translation. More detail on:
//      https://en.wikipedia.org/wiki/Affine_transformation#/media/File:2D_affine_transformation_matrix.svg
//
//       | s*cos, -s*sin, (-s*cos + s*sin) * patchSize/2 + keypoint-x   |
//   M = | s*sin,  s*cos, (-s*sin - s*cos) * patchSize/2 + keypoint-y   |
//       | not  ,  used , here                                          |
//
//   s is the scale-factor and is keypointsize / patchsize
//   if the keypointsize is 0, s should be 1
//   The result of the transformation should be stored in the patch-matrix and should be patchsize * pathcsize
//
//   Angle of Keypointis in degree and is -1 if not given (default = 0)
//
// Parameters:
// const Mat& img:       The image where the patch around the keypoint should be cut out
// const KeyPoint& kp:   The keypoint, which should be the center of the patch
// const int& patchSize: The size of the resulting patch
// Mat& patch:           The patch which is cut out from the image
//==============================================================================
void rectifyPatch(const Mat& img,
                  const KeyPoint& kp,
                  const int& patchSize,
                  Mat& patch)
{
    int flags = CV_WARP_INVERSE_MAP + CV_INTER_CUBIC + CV_WARP_FILL_OUTLIERS;

    Mat transformation_mat = Mat::zeros(2, 3, CV_32FC1);

    float scale;
    if (kp.size <= 0.0f) {
        scale = 1.0f;
    }
    else
    {
        scale = kp.size/float(patchSize);
    }


    float theta;
    if(kp.angle < 0.0f)
    {
        theta = 0.0f;
    }
    else
    {
        theta = kp.angle;
    }
    theta *= (M_PI / 180.0f);

    transformation_mat.at<float>(0,0) = std::cos(theta) * scale;
    transformation_mat.at<float>(1,0) = std::sin(theta) * scale;
    transformation_mat.at<float>(0,1) = std::sin(theta) * (-1 * scale);
    transformation_mat.at<float>(1,1) = std::cos(theta) * scale;
    transformation_mat.at<float>(0,2) = (std::cos(theta) * (-1 * scale) + std::sin(theta) * scale) * patchSize / 2 + kp.pt.x;
    transformation_mat.at<float>(1,2) = (std::sin(theta) * (-1 * scale) - std::cos(theta) * scale) * patchSize / 2 + kp.pt.y;

    cv::warpAffine(img, patch, transformation_mat, cv::Size(patchSize, patchSize), flags);
}


//==============================================================================
// float computeWLResponse(const WeakLearner& WL,
//                         const int orientQuant,
//                         const vector<Mat>& patchIntegrals)
//------------------------------------------------------------------------------
// Computes the response of one WeakLearner for a given patch of the image.
//
// TODO:
// - compute the sum of the gradients in the WeakLearners direction (WL.orient) within a rectangular part of the current patch by
//   using the integral image of these gradients in the given patch (integrals of gradients are in the patchIntegrals vector)
//   (Check out the purpose of integral images if you are unsure what to do)
// - The upper left and lower right points of the rectangular part are Point(WL.x_min, WL.y_min) and Point(WL.x_max, WL.y_max).
// - Calculate the same sum in the total integral image, which is the last one in the patchIntegrals vector (index orientQuant).
// - The weak learner response is now the ratio between the first and the second calculated rectangular sum minus the threshold WL.thresh,
//   but only if the second sum is positive. Otherwise the response is zero.
// - Return this response.
//
// Parameters:
// const vector<Mat>& WL: The WeakLearner for which the response shall be calculated
// const int orientQuant: The number of directions, in which the gradients where calculated
// vector<Mat>& patchIntegrals: The vector of integral images of the gradients (+ sum of these integral images)
//==============================================================================
float computeWLResponse(const WeakLearner& WL,
                        const int orientQuant,
                        const vector<Mat>& patchIntegrals)
{
    float down = patchIntegrals.at(orientQuant).at<int>(WL.y_min, WL.x_min);
    down = down + patchIntegrals.at(orientQuant).at<int>(WL.y_max + 1, WL.x_max + 1);
    down = down - patchIntegrals.at(orientQuant).at<int>(WL.y_max + 1, WL.x_min);
    down = down - patchIntegrals.at(orientQuant).at<int>(WL.y_min, WL.x_max + 1);

    if(down == 0.0f)
    {
        return down;
    }

    float up = patchIntegrals.at(WL.orient).at<int>(WL.y_min, WL.x_min);
    up = up + patchIntegrals.at(WL.orient).at<int>(WL.y_max + 1, WL.x_max + 1);
    up = up - patchIntegrals.at(WL.orient).at<int>(WL.y_max + 1, WL.x_min);
    up = up - patchIntegrals.at(WL.orient).at<int>(WL.y_min, WL.x_max + 1);

    float response = (up / down) - WL.thresh;
    return response;
}



//==============================================================================
// void computeBinboostDescriptorsRectify(const Mat& image,
//                                 const BinBoostProperties& properties,
//                                 const vector<KeyPoint>& features,
//                                 Mat& descriptors)
//------------------------------------------------------------------------------
// Computes the descriptors using binBoost for the previous detected features.
//
// TODO:
// - do the following for each feature:
// - compute the patch of the image around the feature point using the rectifyPatch-function
// - then compute the gradients and integrals of that patch using the compute-functions
//
// - Now compute the responses of the weak learners:
// - For each bit d in the feature descriptor (0 <= d < properties.nDim) calculate the properties.nWLs weak learner responses:
// - start off with response = 0
// - For each weak learner (0 <= wl < properties.nWLs) add + or - properties.pWLsArray[d][wl].beta to the response,
//   depending on the sign of the return value of the computeWLResponse function with the parameters
//   current wl properties from the properties.pWLsArray array at dimensions d and weaklearner number wl, and the number of directions (properties.orientQuant)
//   and the patchIntegrals.
//   if that return value is positive, add the beta value, otherwise subtract it from the total response.
// - For each feature there is one row in the descriptors Mat. And in each row there are properties.nDim bits to be set.
// - Whether the d-th bit shall be set or not depends on the sign of the total weak learners response. for positive response,
//   the bit has to be set to the according (d modulo 8)-th bit in the properties.binLookUp array, otherwise it shall be set to zero.
//
//
// Parameters:
// const Mat& img: the grayscale input image
// const BinBoostProperties& properties: properties of the binboost feature description
// const vector<KeyPoint>& features: previous detected features
// Mat& descriptors: Mat to be filled with the computed descriptors
//==============================================================================
void computeBinboostDescriptorsRectify(const Mat& img,
                                const BinBoostProperties& properties,
                                const vector<KeyPoint>& features,
                                Mat& descriptors)
{
    // initialize the variables
    descriptors = Mat::zeros(features.size(), ceil(properties.nDim/8), CV_8UC1);
    vector<Mat> patchIntegrals, patchGradients;

    // iterate through all the features
    for (unsigned int i=0; i<features.size(); ++i)
    {
        // rectify the patch around a given keypoint
        Mat patch;
        rectifyPatch(img, features[i], properties.patchSize, patch);
        // if rectifyPatch is not implemented, patch should be empty.
        if (patch.empty())
            continue;


        Mat dir_tmp, mag_tmp;
        computeGradients(patch, properties, patchGradients, dir_tmp, mag_tmp);
        computeIntegrals(patchGradients, properties.orientQuant, patchIntegrals);

        // compute the responses of the weak learners
        float resp;

        for (int d = 0; d < properties.nDim; d++)
        {
            resp = 0;
            for (int wl = 0; wl < properties.nWLs; wl++)
            {
                resp += (computeWLResponse(properties.pWLsArray[d][wl],
                                           properties.orientQuant, patchIntegrals) >= 0) ?
                        properties.pWLsArray[d][wl].beta : -properties.pWLsArray[d][wl].beta;
            }
            descriptors.at<uchar>((int)i, (int)(d/8)) |= (resp >= 0) ? properties.binLookUp[d%8] : 0;
        }

        // clean-up
        patch.release();
        patchIntegrals.clear();
        patchGradients.clear();
    }
}




//==============================================================================
// void buildOctavesPyramid(const Mat& img,
//                          const int numOctaves,
//                          const int numLayers,
//                          vector<vector<Mat>>& pyramid,
//                          vector<vector<Mat>>& DOGpyramid)
//------------------------------------------------------------------------------
// Build the pyramid of differently smoothed images at each layer
// and different sizes at each octave.
//
// TODO:
// - compute a the first gaussian blurred layer from img.
//   Use this formula for the kernel_size: kernel_size = ceil(sigma * 3) * 2 + 1
// - put the blurred image into the layers vector.
// - Now do the following for each octave from 0 to numOctaves - 1:
//  - for each layer from 1 (the 0th layer is already pushed) to numLayers + 2:
//  - use the formula for sigma from the assignment sheet and the same formula for
//    the kernel_size as above.
//  - Compute the Gaussian blured image with above parameters from the previous inserted layer.
//  - Insert that new blured layer into the layers vector.
//  - compute the absolute difference between the new and the previous layer (difference of gaussian)
//    and insert it into the DOG_layers vector.
//  - after calculating all layers, push the layers and DOG_layers to the corresponding pyramids.
//  - Now we have to resize the image for the new ovcave. take the numLayers-th image in the layers vector
//    and halve it in width and height.
//
//
//  useful opencv functions: GaussianBlur, absdiff, resize (INTER_AREA)
//
//
// Parameters:
// const Mat& img: the grayscale input image
// const int numOctaves: number of octaves (depth of pyramid)
// const int numLayers: number of layers per octave
// vector<vector<Mat>>& pyramid: the pyramid of octaves to be filled. (size: numOctaves x (numLayers + 3))
// vector<vector<Mat>>& DOGpyramid: the pyramid of diffs of previous and current layer (size: numOctaves x (numLayers + 2))
//==============================================================================
void buildOctavesPyramid(const Mat& img,
                         vector<vector<Mat>>& pyramid,
                         vector<vector<Mat>>& DOGpyramid)
{
    const float sigma0 = 1;
    const float k = powf(2, 1/float(numLayers));

    int kernel_size = 0;
    vector<Mat> DOG_layers, layers;
    float sigma = sigma0;

    kernel_size = cvCeil(sigma * 3) * 2 + 1;
    Mat first_layer;
    GaussianBlur(img, first_layer, cv::Size(kernel_size, kernel_size), sigma0, sigma0);
    layers.push_back(first_layer);

    int current_octave = 0;
    while (current_octave < numOctaves)
    {
        float previous_sigma = sigma;
        for (int current_layer = 1; current_layer <= numLayers + 2; current_layer++)
        {
            float sigma_l = sigma0 * pow(k, current_layer);
            float sigma_dach = sqrt(pow(sigma_l, 2) - pow(previous_sigma, 2));
            previous_sigma = sigma_l;

            kernel_size = cvCeil(sigma_dach * 3) * 2 + 1;

            Mat following_layer, DOG;

            GaussianBlur(layers.at(current_layer - 1), following_layer, cv::Size(kernel_size, kernel_size), sigma_dach, sigma_dach);
            layers.push_back(following_layer);

            absdiff(layers.at(current_layer), layers.at(current_layer - 1), DOG);
            DOG_layers.push_back(DOG);
        }

        pyramid.push_back(layers);
        DOGpyramid.push_back(DOG_layers);

        Mat resized;
        resize(layers.at(numLayers), resized, cv::Size(), 0.5, 0.5, INTER_AREA);

        layers.clear();
        DOG_layers.clear();
        layers.push_back(resized);

        current_octave++;
    }
}

//==============================================================================
// void calculateHarrisCornerMeasure(const Mat& current_layer,
//                                   const float sigma_i,
//                                   const float sigma_d,
//                                   Mat& cornerness_mat)
//------------------------------------------------------------------------------
// Calculate the Harris Corner Measure for a layer and given sigma values for bluring.
//
// TODO:
// - compute the (approximated) derivatives of the layer in x and y direction using the Sobel operator.
// - Multiply the derivatives with sigma_d, the differentiation scale.
// - compute the squared derivatives images in x and y direction and the mixed derivative Lxy.
// - calculate the kernel_size for smoothing with sigma_i as defined in the assignment sheet
// - smooth L_xx, L_yy L_xy, using the integration scale sigma_i. (Averaging in the neighborhood)
// - Now we can calculate the cornerness_mat. The cornerness response at image location x, y is defined as follows:
//   R = det(H) - k·trace(H))², where H is the 2x2 matrix [dxx dxy; dxy dyy] and dxx, dxy, dyy the (smoothed) second derivatives of the image at the same location.
//
//
// Parameters:
// const Mat& current_layer: the current layer in the octave pyramid
// const float sigma_i: "integration scale" - used for bluring the image
// const float sigma_d: "differentiation scale" - used for normalizing the Derivatives
// Mat& cornerness_mat: Matrix that shall be filled with the cornerness response at the image locations.
//==============================================================================
void calculateHarrisCornerMeasure(const Mat& current_layer,
                                  const float sigma_i,
                                  const float sigma_d,
                                  Mat& cornerness_mat)
{
    Mat Lx, Ly, Lxx, Lyy, Lxy;
    Mat LxxSmooth, LyySmooth, LxySmooth;

    int kernel_size = 0;

    cornerness_mat = Mat::zeros(current_layer.size(), CV_32F);

    Sobel(current_layer, Ly, CV_32F, 0, 1, 1, BORDER_DEFAULT);
    Sobel(current_layer, Lx, CV_32F, 1, 0, 1, BORDER_DEFAULT);

    Ly = Ly.mul(sigma_d);
    Lx = Lx.mul(sigma_d);
    multiply(Lx, Ly, Lxy);
    multiply(Lx, Lx, Lxx);
    multiply(Ly, Ly, Lyy);

    kernel_size = cvCeil(sigma_i * 3) * 2 + 1;
    GaussianBlur(Lxy, LxySmooth, cv::Size(kernel_size, kernel_size), sigma_i, sigma_i, BORDER_REPLICATE);
    GaussianBlur(Lyy, LyySmooth, cv::Size(kernel_size, kernel_size), sigma_i, sigma_i, BORDER_REPLICATE);
    GaussianBlur(Lxx, LxxSmooth, cv::Size(kernel_size, kernel_size), sigma_i, sigma_i, BORDER_REPLICATE);

    for (int current_row = 0; current_row < current_layer.rows; current_row++)
    {
        for (int current_col = 0; current_col < current_layer.cols; current_col++)
        {
            float lxx_value = LxxSmooth.at<float>(current_row, current_col);
            float lyy_value = LyySmooth.at<float>(current_row, current_col);
            float lxy_value = LxySmooth.at<float>(current_row, current_col);

            float det = lxx_value * lyy_value - pow(lxy_value, 2);

            float trace_H = lxx_value + lyy_value;
            cornerness_mat.at<float>(current_row, current_col) = det - 0.04f * powf(trace_H, 2);
        }
    }
}

//==============================================================================
// void nonMaximaSuppressionHarris(const cv::Mat& cornerness,
//                                 cv::Mat& local_maxima)
//------------------------------------------------------------------------------
// Performs non-maxima suppression on the cornerness Matrix ``cornerness'' and
// stores it in the local_maxima Matrix.
//
// TODO:
// - Compute the maximum value of the cornerness matrix
// - Set all entries in the cornerness measure smaller than 
//   maxVal * cornerness_threshold to 0 (=cornerness_thresholded).
// - Dilate the thresholded cornerness image with a 3x3 matrix (=cornerness_dilate).
// - The resulting local maximas are the non-zero entries of the dilated image,
//   which are equal to the cornerness_threshold entries.
//   (cornerness_dilate == cornerness_thresholded) && (cornerness_thresholded > 0)
//
// Parameters:
// const cv::Mat &cornerness: The cornerness measure obtained from the harris
//                            corner detector.
// cv::Mat &local_maxima: The local maxima of the harris corner detection.
// cornerness_threshold: The threshold for the cornerness
void nonMaximaSuppressionHarris(const cv::Mat &cornerness, cv::Mat &local_maxima, 
        float cornerness_threshold=0.01)
{
    local_maxima = cv::Mat::zeros(cornerness.size(), CV_8UC1);
    // - Compute the maximum value of the cornerness matrix
    double max, min;
    minMaxLoc(cornerness, &min, &max);

    Mat thresholded;
    threshold(cornerness, thresholded, max * cornerness_threshold, max, THRESH_TOZERO);

    Mat dilated;
    dilate(thresholded, dilated, Mat());

    Mat dt, t;
    compare(thresholded, dilated, dt, CMP_EQ);
    compare(thresholded, 0, t, CMP_GT);
    bitwise_and(dt, t, local_maxima);
}


bool isInside(float x_coord, float y_coord, float size, Mat img) {
    return x_coord + size / 2.0f < img.cols
           && y_coord + size / 2.0f < img.rows
           && x_coord - size / 2.0f >= 0
           && y_coord - size / 2.0f >= 0;
}

bool checkMaximumAndThreshold(vector<vector<Mat>> &DOGpyramid, int current_octave, int current_layer, float DOG_thresh, int current_row, int current_column) {
    Mat dog_layer_current = DOGpyramid.at(current_octave).at(current_layer);
    Mat dog_layer_previous = DOGpyramid.at(current_octave).at(current_layer-1);
    Mat dog_layer_following = DOGpyramid.at(current_octave).at(current_layer+1);

    return dog_layer_current.at<uint8_t>(current_row,current_column) > dog_layer_previous.at<uint8_t>(current_row,current_column)
           && dog_layer_current.at<uint8_t>(current_row,current_column) > dog_layer_following.at<uint8_t>(current_row,current_column)
           && dog_layer_current.at<uint8_t>(current_row,current_column) >= DOG_thresh;
}

//==============================================================================
// void detectFeatures(const Mat& img,
//                     vector<KeyPoint>& features,
//                     vector<vector<Mat>> &pyramid,
//                     vector<vector<Mat>> &DOGpyramid)
//                     )
//------------------------------------------------------------------------------
// Detect feature points in the image using Harris Laplace Feature Detection
// as described in this paper:
// https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_ijcv2004.pdf
//
// TODO:
// - go through all octaves and their layers from 1 to numLayers.
// - for octave = 0 start with layer = 2, as the previous octave does not exist. 
//   If the octave >= 1, start at layer = 1.
// - get the current_layer: this is either the layer with number (layer - 2), if layer > 1,
//   or the (numLayers - 1)-th layer in the previous octave.
//   in the second case, you have to resize the image by halving its size.
// - calculate sigma_i and sigma_d according to the formulas given in the assignment sheet.
// - using these sigma values, compute the cornerness Mat using the function calculateHarrisCornerMeasure.
// - now we have to perform a Non-Max-Suppression on the cornerness Mat (nonMaximaSuppressionHarris).
// - for all non-maximas get the DoG values at the previous, current and following DOG layers at the current octave.
// - calculate the potential keypoint with these parameters:
//      x = (2^{octave}) * column
//      y = (2^{octave}) * row
//      the size is defined as 3 * 2^{octave} * sigma_i * 2 (see document)
//      the angle should be set to 0
//      for the resonse should be the cornerness response
//      the octave should be set to octave * (numLayers + 1) + layer (!important)
// - that keypoint is added to the features vector, if the whole keypoint is within the image bounds and
//   the current DoG value is greater than previous and following, and also greater or equal to the DOG_thresh constant.
//
//
// Parameters:
// const Mat& img: the grayscale input image
// vector<KeyPoint>& features: the vector of featurepoints to be filled
// vector<vector<Mat>>& pyramid: the gaussian scale space pyramid
// vector<vector<Mat>>& dog_pyramid: the dog space pyramid
//==============================================================================
void detectFeatures(const Mat& img,
                    vector<KeyPoint>& features,
                    vector<vector<Mat>> &pyramid,
                    vector<vector<Mat>> &DOGpyramid) {
    const float DOG_thresh = 1e-40;
    const float cornerness_threshold = 0.01;

    float sigma_i = 0, sigma_d = 0;

    DOGpyramid.clear();
    pyramid.clear();
    buildOctavesPyramid(img, pyramid, DOGpyramid);


    for (int current_octave = 0; current_octave < numOctaves; current_octave++)
    {
        for (int current_layer = 1; current_layer <= numLayers; current_layer++)
        {
            Mat current_layer_mat;
            if (current_octave == 0 && current_layer == 1)
            {
                continue;
            }
            else if (current_layer >= 2)
            {
                current_layer_mat = pyramid.at(current_octave).at(current_layer - 2);
            }
            else if (current_layer == 1 && current_octave >= 1)
            {
                Mat source = pyramid.at(current_octave - 1).at(numLayers - 1);
                cv::resize(source, current_layer_mat, cv::Size(), 0.5f, 0.5f, INTER_AREA);
            }

            sigma_i = powf(2.0f, (float) current_layer / (float) numLayers);
            sigma_d = sigma_i * 0.7f;

            Mat cornerness_mat, non_maxima_mat;
            calculateHarrisCornerMeasure(current_layer_mat, sigma_i,sigma_d, cornerness_mat);
            nonMaximaSuppressionHarris(cornerness_mat, non_maxima_mat, cornerness_threshold);

            for (int current_row = 1; current_row < non_maxima_mat.rows; current_row++)
            {
                for (int current_column = 1; current_column < non_maxima_mat.cols; current_column++)
                {
                    if (non_maxima_mat.at<uint8_t>(current_row, current_column) > 0)
                    {
                        float size = 3.0f * pow(2,current_octave) * sigma_i * 2.0f;
                        float response = cornerness_mat.at<float>(current_row,current_column);
                        float angle = 0.0f;
                        int octave = current_octave * (numLayers + 1) + current_layer;
                        float x_coord = powf(2.0f, current_octave) * current_column;
                        float y_coord = pow(2.0f, current_octave) * current_row;

                        if (isInside(x_coord, y_coord, size, img) &&
                            checkMaximumAndThreshold(DOGpyramid, current_octave, current_layer, DOG_thresh, current_row, current_column))
                        {
                            features.emplace_back(cv::KeyPoint(x_coord, y_coord, size, angle, response, octave));
                        }
                    }
                }
            }
        }
    }
}



//==============================================================================
// binBoostFeaturesHarrisLaplace(const Mat& img,
//                      const BinBoostProperties& properties,
//                      vector<vector<Mat>>& pyramid,
//                      vector<vector<Mat>>& dog_pyramid,
//                      vector<Mat>& gradients,
//                      vector<Mat>& integrals,
//                      Mat directions,
//                      Mat magnitudes,
//                      vector<KeyPoint>& features,
//                      Mat& descriptors)
//------------------------------------------------------------------------------
// Find BinBoost feature points and descriptors
//
// TODO:
// - convert the three channel color image to grayscale
// - determine the feature points using the function detectFeatures.
// - compute the gradients in each direction using the function computeGradients
// - compute the gradientIntegrals in each direction using the function computeIntegrals
// - compute the feature descriptors using the function computeBinboostDescriptorsRectify.
//          You will also need computeGradients and computeIntegrals.
//
//
// Parameters:
// const Mat& img: the input image (three channel)
// const BinBoostProperties& properties: properties of the binboost feature descriptor
// vector<vector<Mat>>& pyramid: the scale space pyramid
// vector<vector<Mat>>& dog_pyramid: the dog pyramid
// vector<Mat>& gradients: vector of Mats that shall be filled with the gradients of the image in each direction
// Mat& directions: Mat that shall be filled with the direction of the gradients
// Mat& magnitudes: Mat that shall be filled with the magnitudes of the gradients
// vector<Mat>& integrals: vector of Mats that shall be filled with the integrals of the gradients in each direction
// vector<KeyPoint>& features: shall be filled with the detected feature points
// Mat& descriptors: shall be flled with the descriptors of the detected features
//==============================================================================
void binBoostFeaturesHarrisLaplace(const Mat& img,
                      const BinBoostProperties& properties,
                      vector<vector<Mat>> &pyramid,
                      vector<vector<Mat>> &dog_pyramid,
                      vector<Mat>& gradients,
                      vector<Mat>& integrals,
                      Mat& directions,
                      Mat& magnitudes,
                      vector<KeyPoint>& features,
                      Mat& descriptors)
{
    Mat grey;

    cvtColor(img, grey, CV_BGR2GRAY);
    
    pyramid.clear();
    dog_pyramid.clear();
    detectFeatures(grey, features, pyramid, dog_pyramid);

    std::vector<std::vector<cv::Mat>> magnitudesPyr, orientationsPyr;

    magnitudesPyr.resize(pyramid.size()); orientationsPyr.resize(pyramid.size());
    for (size_t o = 0; o < pyramid.size(); o++)
    {
        magnitudesPyr[o].resize(pyramid[o].size()); orientationsPyr[o].resize(pyramid[o].size());
        for (size_t l = 0; l < pyramid[o].size(); l++)
        {
            computeGradientOrientationsAndMagnitudes(pyramid[o][l], orientationsPyr[o][l], magnitudesPyr[o][l]);
        }
    }

    // write out for debug purposes only
    computeGradients(grey, properties, gradients, directions, magnitudes);
    computeIntegrals(gradients, properties.orientQuant, integrals);

    computeOrientation(features, properties, 36, orientationsPyr, magnitudesPyr);
    computeBinboostDescriptorsRectify(grey, properties, features, descriptors);
}

//==============================================================================
// binBoostFeaturesFast(const Mat& img,
//                          const BinBoostProperties& properties,
//                          vector<Mat>& gradients,
//                          vector<Mat>& integrals,
//                          Mat& directions,
//                          Mat& magnitudes,
//                          vector<KeyPoint>& features,
//                          Mat& descriptors,
//                          const vector<Point> &circle)
//------------------------------------------------------------------------------
// Find BinBoost feature points and descriptors
//
// TODO: (BONUS TASK)
// - convert the three channel color image to grayscale
// - compute the harris measure of the grayscale image in the Mat R using the function calculateHarrisCornerMeasure
//   with sigma_i = 0 (no smoothing in neighborhood) and sigma_d = 1.
// - perform a non-max-suppression similar to the one in the function detectFeatures
// - determine the feature points using the function Fast with the given constant values and the cornerness mat.
// - compute the gradients in each direction using the function computeGradients
// - compute the gradientIntegrals in each direction using the function computeIntegrals
// - compute the feature descriptors using the function computeBinboostDescriptors.
//          You will also need computeGradients and computeIntegrals.
//
//
// Parameters:
// const Mat& img: the input image (three channel)
// const BinBoostProperties& properties: properties of the binboost feature descriptor
// vector<Mat>& gradients: vector of Mats that shall be filled with the gradients of the image in each direction
// vector<Mat>& integrals: vector of Mats that shall be filled with the integrals of the image in each direction
// Mat& directions: Mat that shall be filled with the direction of the gradients
// Mat& magnitudes: Mat that shall be filled with the magnitudes of the gradients
// vector<KeyPoint>& features: shall be filled with the detected feature points
// Mat& descriptors: shall be flled with the descriptors of the detected features
// const vector<Point>& circle: Points on a circle with radius 3.
//==============================================================================
void binBoostFeaturesFast(const Mat& img,
                          const BinBoostProperties& properties,
                          vector<Mat>& gradients,
                          vector<Mat>& integrals,
                          Mat& directions,
                          Mat& magnitudes,
                          vector<KeyPoint>& features,
                          Mat& descriptors,
                          const vector<Point> &circle)
{
    Mat R;
    Mat grey;
    
    const int N = 12;
    const int thresh = 15;
    const float harrisThreshold = 0.1;
    
    cvtColor(img, grey, CV_BGR2GRAY);
    
    cv::Mat localMax;
    calculateHarrisCornerMeasure(grey, 1, 1, R);
    nonMaximaSuppressionHarris(R, localMax);
    localMax.convertTo(localMax, CV_32FC1, 1.f/255.f); 
    R = R.mul(localMax);
    
    Fast(grey, features, circle, R, N, thresh, harrisThreshold);

    computeGradientOrientationsAndMagnitudes(grey, directions, magnitudes);
    std::vector<std::vector<cv::Mat>> directionsPyr(1);
    std::vector<std::vector<cv::Mat>> magnitudesPyr(1);
    directionsPyr[0].push_back(directions);
    magnitudesPyr[0].push_back(magnitudes);

    computeOrientation(features, properties, 36, directionsPyr, magnitudesPyr);
    computeBinboostDescriptorsRectify(grey, properties, features, descriptors);

}

//==============================================================================
// readWLsBinForBinboost(BinBoostProperties& properties)
//------------------------------------------------------------------------------
// Load binboost descriptor properties from binboost binary file including the
// learned parameters of the Weak Learner (WL).
//
// TODO:
//	- Nothing!
//	- Do not change anything here
//
// Parameters:
// BinBoostProperties& properties: properties of the binboost feature descriptor
//                                 to be loaded
//==============================================================================
void readWLsBinForBinboost(BinBoostProperties& properties)
{
    FILE* bin_file = fopen(properties.wl_bin_path.c_str(),"rb");
    if (!bin_file)
    {
        fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", properties.wl_bin_path.c_str());
        exit(-3);
    }
    if(fread(&properties.nDim,sizeof(int),        1,bin_file) != 1) exit(-1);
    if(fread(&properties.orientQuant,sizeof(int), 1,bin_file) != 1) exit(-1);
    if(fread(&properties.patchSize,sizeof(int),   1,bin_file) != 1) exit(-1);
    int iGradAssignType;
    if(fread(&iGradAssignType,sizeof(int), 1,bin_file) != 1) exit(-1);
    properties.gradAssignType = (Assign) iGradAssignType;
    for (int i = 0; i < 8; i++)
        properties.binLookUp[i] = (char) 1 << i;

    properties.pWLsArray = new WeakLearner*[properties.nDim];
    for (int d = 0; d<properties.nDim; ++d)
    {
        if(fread(&properties.nWLs,sizeof(int), 1,bin_file) != 1) exit(-1);
        properties.pWLsArray[d] = new WeakLearner[properties.nWLs];
        for (int i = 0;i < properties.nWLs;i++)
        {
            if(fread(&properties.pWLsArray[d][i].thresh,sizeof(float), 1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].orient,sizeof(int),   1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].y_min,sizeof(int),    1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].y_max,sizeof(int),    1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].x_min,sizeof(int),    1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].x_max,sizeof(int),    1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].alpha,sizeof(float),  1,bin_file) != 1) exit(-1);
            if(fread(&properties.pWLsArray[d][i].beta,sizeof(float),   1,bin_file) != 1) exit(-1);
        }
    }
    fclose(bin_file);
}


//==============================================================================
// matchFeatures(const vector<KeyPoint>& features, const Mat& descriptors,
//     const vector<KeyPoint>& new_features, const Mat& new_descriptors,
//     vector<Point2f>& good_matches_source, vector<Point2f>&
//     good_matches_target, DescriptorMatcher& matcher)
//------------------------------------------------------------------------------
// Calculates matches between source and target feature descriptors
//
// TODO:
// - calculate matches between source_descriptors and target_descriptors by
//   calling matcher.knnMatch(...)
// - discard matches where the distance of the first match is less than
//   match_distance_factor times distance of the second match
// - insert the remaining features into the output variables good_matches_source
//   and good_matches_target by using DMatch::queryIdx and DMatch::trainIdx of
//   the first entry of each match pair to index the KeyPoint from
//   source_features and target_features
//
// Parameters:
// const vector<KeyPoint>& source_features: Features of the source image
// const Mat& source_descriptors: Descriptor vectors of the source image
// const vector<KeyPoint>& target_features: Features of the target image
// const Mat& target_descriptors: Descriptor vectors of the target image
// vector<Point2f>& good_matches_source: Points in the source image that have a
//                                       correspondence in the target image.
// vector<Point2f>& good_matches_target: Points in the target image that have a 
//                                       correspondence in the source image.
// const DescriptorMatcher& matcher: Used for finding matches between
//                                   source_descriptor and target_descriptor
//==============================================================================
void matchFeatures(const vector<KeyPoint>& source_features,
                   const Mat& source_descriptors,
                   const vector<KeyPoint>& target_features,
                   const Mat& target_descriptors,
                   const float match_distance_factor,
                   vector<DMatch>& matches,
                   vector<Point2f>& good_matches_source,
                   vector<Point2f>& good_matches_target,
                   const DescriptorMatcher& matcher) {

    const int k = 2;

    std::vector<std::vector<DMatch>> knns;
    matcher.knnMatch(source_descriptors, target_descriptors, knns, k);

    for (int current_match = 0; current_match < knns.size(); current_match++)
    {
        float first_knn_distance = knns.at(current_match).at(0).distance;
        float second_knn_distance = knns.at(current_match).at(1).distance;
        if(first_knn_distance <= match_distance_factor * second_knn_distance)
            matches.push_back(knns.at(current_match).at(0));
    }

    for (auto & iterator : matches)
    {
        good_matches_source.push_back(source_features.at(iterator.queryIdx).pt);
        good_matches_target.push_back(target_features.at(iterator.trainIdx).pt);
    }
}

//==============================================================================
// Mat calculateHomography(const vector<Point2f>& good_matches_source,
//     const vector<Point2f>& good_matches_target
//------------------------------------------------------------------------------
// Calculates a homography between a number of point correspondences
//
// TODO:
// - Calculate the optimal homography between good_matches_source and
//   good_matches_target by using the OpenCV function cv::findHomography with
//   the parameter RANSAC and ansacReprojThreshold=3
// - Convert the calculated homography to CV_32F
//
// parameters:
// const vector<Point2f>& good_matches_source: Points in the source image that
//                                             have a correspondence in the 
//                                             target image 
// const vector<Point2f>& good_matches_target: Points in the target image that 
//                                             have a correspondence in the 
//                                             source image
//
// return: 3x3 matrix which best approximates the transformation from
//         good_matches_source to good_matches_target
//==============================================================================
Mat calculateHomography(const vector<Point2f>& good_matches_source,
                        const vector<Point2f>& good_matches_target)
{
    if (good_matches_source.size() != 0 && good_matches_target.size() != 0)
    {
        Mat homography, homography_in_CV32F;
        homography = cv::findHomography(good_matches_source,good_matches_target, CV_RANSAC,3);
        homography.convertTo(homography_in_CV32F, CV_32F);
        return homography_in_CV32F;
    }
    else
    {
        return Mat::eye(3,3,CV_32F);
    }
}




//==============================================================================
// void computeInliers(const vector<DMatch>  matches,
//                     const cv::Mat& homography,
//                     const std::vector<Point2f>& good_matches_left,
//                     const std::vector<Point2f>& good_matches_right,
//                     std::vector< bool>& inliers)
//------------------------------------------------------------------------------
// Calculate the good and bad matches based on the ransac inliers.
//
// TODO: nothing to do here
//
// parameters:
// const vector<DMatch>&  matches:             the matches incl. their distance
// std::vector<Point2f>& homography:   homography matrix
// const std::vector<Point2f>& good_matches_left:   Points of the matches in the left image
// const std::vector<Point2f>& good_matches_right:  Points of the matches in the right image
// std::vector<bool>& inliers:  inliers flag
//==============================================================================
void computeInliers(const vector<DMatch>& matches,
                    const Mat &homography,
                    const std::vector<Point2f>& good_matches_left,
                    const std::vector<Point2f>& good_matches_right,
                    std::vector<bool> &inliers)
{

    cv::Mat src_points(3, matches.size(), CV_32FC1);
    for (int i = 0; i < matches.size(); i++)
    {
        src_points.at<float>(0, i) = good_matches_left[i].x;
        src_points.at<float>(1, i) = good_matches_left[i].y;
        src_points.at<float>(2, i) = 1.f;
    }

    cv::Mat dst_points = homography * src_points;

    inliers.resize(matches.size(), false);
    for (size_t i = 0; i < matches.size(); i++)
    {
        dst_points.at<float>(0, i) /= std::max(1e-5f, dst_points.at<float>(2, i));
        dst_points.at<float>(1, i) /= std::max(1e-5f, dst_points.at<float>(2, i));

        float dx = dst_points.at<float>(0, i) - good_matches_right[i].x;
        float dy = dst_points.at<float>(1, i) - good_matches_right[i].y;

        inliers[i] = std::sqrt(dx * dx - dy * dy) < 3;
    }
}





//==============================================================================
//void transformImage(const Mat& img_right,
//                    const Mat& img_left,
//                    const Mat& homography,
//                    Mat& img_right_transformed)
//------------------------------------------------------------------------------
// Transform the img_right with the given homography.
//
// TODO:
// - The homography estimates the transform from the left image to the right image.
// - Use cv::warpPerspective to rectify the right image to match the left one
// - The output size should be (img_left.cols, img_left.rows)
// - Use linear interpolation in cv::warpPerspective
//
// parameters:
// const Mat& img_right:           The image that has to be transformed
// const Mat& img_left:            The image to compare it in a later step (used for the correct size)
// const Mat& homography:          A homography which transforms the left image to the right.
// Mat& img_right_transformed:     The transformed image
//==============================================================================
void transformImage(
    const cv::Mat &img_right,
    const cv::Mat &img_left,
    const cv::Mat &homography,
    cv::Mat &img_right_transformed)
{
    cv::warpPerspective(img_right, img_right_transformed, homography, cv::Size(img_left.cols, img_left.rows), INTER_LINEAR + WARP_INVERSE_MAP);
}




//==============================================================================
//void computeDiffs(  const Mat& img_left,
//                    const Mat& img_right,
//                    const float sigma,
//                    const double threshold,
//                    Mat& img_diff,
//                    Mat& img_threshold,
//                    vector<vector<Point>>& contours)
//------------------------------------------------------------------------------
// Calculate the difference between two images
//
// TODO:
// - For pre-processing, we blur both images with a Gaussian filter with the given ``sigma'' parameter.
// - The difference of the blurred same-sized images can be calculated using the simple absdiff-function.
// - For simplicity we convert the difference image to a grayscale image, to get the average difference for a given
//   pixel.
// - Then a threshold is done using the diff-image which was previously converted to greyscale and stored in diffImg.
// - To find and mark the differences we usa a Canny-Edge-Filter in combination with findContours() in the diff-image.
//       Canny():        The threshold was done before and has not to be specified. The aperture-size should be 3)
//       findContours(): following options are used: RETR_TREE and CHAIN_APPROX_SIMPLE  (with offset [0, 0])
//
// parameters:
// const Mat& img_left:              First image for the difference calculation
// const Mat& img_right:             Second image
// const float sigma:                Is used for blurring the image
// const double threshold:           Is used to threshold the image
// Mat& diffImg:                     The difference-image
// Mat& img_threshold:               The threholded difference-image
// vector<vector<Point>>& contours): The contours found with the Canny-Edge
//==============================================================================
void computeDiffs(  const Mat& img_left,
                    const Mat& img_right,
                    const float sigma,
                    const double threshold,
                    Mat& diffImg,
                    Mat& img_threshold,
                    vector<vector<Point>>& contours)
{

    Mat cannyOutput, left_image_blurred, right_image_blurred, diff_in_grayscale;

    vector<Vec4i> hierarchy;

    int size = cvCeil(3 * sigma) * 2 + 1;

    GaussianBlur(img_left, left_image_blurred, cv::Size(size,size), sigma,sigma);

    GaussianBlur(img_right, right_image_blurred, cv::Size(size,size), sigma,sigma);

    absdiff(left_image_blurred, right_image_blurred, diffImg);

    cvtColor(diffImg, diff_in_grayscale, CV_BGR2GRAY);

    cv::threshold(diff_in_grayscale, img_threshold, threshold , 255, THRESH_BINARY);

    Canny(img_threshold,cannyOutput,0,255,3);

    findContours(cannyOutput, contours, hierarchy, RETR_TREE,CHAIN_APPROX_SIMPLE);
}



//==============================================================================
//bool doRectOverlap( Rect& rect1,
//                    Rect& rect2)
//------------------------------------------------------------------------------
// Returns if two rectangles overlap
//
// TODO: nothing to do
// - The function returns true if one of the rectangles is overlapping the other one
//
// parameters:
// const Rect& rect1:  The first rectangle
// const Rect& rect2:  The second one
//==============================================================================
bool doRectOverlap( const Rect& rect1,
                    const Rect& rect2)
{

    Point l1(rect1.x, rect1.y);
    Point l2(rect2.x, rect2.y);
    Point r1(rect1.x + rect1.width, rect1.y + rect1.height);
    Point r2(rect2.x + rect2.width, rect2.y + rect2.height);

    // If one rectangle is on left side of other
    if (l1.x > r2.x || l2.x > r1.x)
        return false;

    // If one rectangle is above other
    if (l1.y > r2.y || l2.y > r1.y)
        return false;

    return true;
}

//==============================================================================
//void markDiffPoints( const vector<vector<Point>>& contours,
//                        const int maxRadius,
//                        vector<Point2f>& centers,
//                        vector<int>& radius,
//                        vector<Rect>& rect_diffs)
//------------------------------------------------------------------------------
// Gets a set of contours and calculates some heatpoints with their bounding rectangle
//
// TODO:
// - For every contour the center point is calculated (arithmetic middle of all x / y)
//   Then the bounding rectangle is calculated with boundingRect() and the diameter is set to the maximum of width/height
//   Both values are saved in the vector centers and radius
// - If the diameter is smaller or equal to maxDiameter, the bounding rect is taken, padded by 10 pixels, and compared 
//   with all rectangles in the diffRectangles-list. If they are overlapping
//   each other the bounding rect should be updated to a new bounding rect which is formed by union of both
//   rectangles. The other rectangle should be erased from the list and the search should start from beginning with
//   the new bounds. If no overlapping rect is found the bounding rect is added to the list.
// - Use the doRectOverlap()-function 
//
// parameters:
// const vector<vector<Point>>& contours:  The given contours
// const int maxDiameter:                  The maximum diameter of a contour
// vector<Point2f>& centers:               A list with the centers of the found heatpoints
// vector<int>& diameters:                 A list with the diameters of the heatpoint-circles
// vector<Rect>& diffRectangles:           The found differences in a list of bounding-rectangles
//==============================================================================
void markDiffPoints( const vector<vector<Point>>& contours,
                     const int maxDiameter,
                     vector<Point2f>& centers,
                     vector<int>& diameters,
                     vector<Rect>& diffRectangles)
{
    // get the centers
    diffRectangles.clear();
    centers.resize(contours.size());
    diameters.resize(contours.size());

    for (int current_contour = 0; current_contour < contours.size(); current_contour++)
    {
        int sum_of_all_x = 0;
        int sum_of_all_y = 0;

        for (int iterator = 0; iterator < contours.at(current_contour).size(); iterator++)
        {
            sum_of_all_x = sum_of_all_x + contours.at(current_contour).at(iterator).x;
            sum_of_all_y = sum_of_all_y + contours.at(current_contour).at(iterator).y;
        }

        float center_point_x = (1 / (float) contours.at(current_contour).size()) * sum_of_all_x;
        float center_point_y = (1 / (float) contours.at(current_contour).size()) * sum_of_all_y;

        centers.at(current_contour) = cv::Point2f(center_point_x, center_point_y);
        Rect rect = cv::boundingRect(contours.at(current_contour));
        diameters.at(current_contour) = max(rect.width, rect.height);

        if (diameters.at(current_contour) <= maxDiameter)
        {
            rect.x = rect.x - 10;
            rect.width = rect.width + 20;

            rect.y = rect.y - 10;
            rect.height = rect.height + 20;

            for (int diff = 0; diff < diffRectangles.size(); diff++)
            {
                if (doRectOverlap(diffRectangles.at(diff), rect))
                {
                    int old_x = rect.x;
                    int old_y = rect.y;
                    rect.x = min(rect.x,diffRectangles.at(diff).x);
                    rect.y = min(rect.y,diffRectangles.at(diff).y);
                    rect.width = max(old_x + rect.width, diffRectangles.at(diff).x + diffRectangles.at(diff).width) - rect.x;
                    rect.height = max(old_y + rect.height, diffRectangles.at(diff).y + diffRectangles.at(diff).height) - rect.y;
                    diffRectangles.erase(diffRectangles.begin() + diff);
                    diff = 0;
                }
            }

            diffRectangles.push_back(rect);
        }
    }
}




//================================================================================
// main()
//--------------------------------------------------------------------------------
// TODO:
//	- Nothing!
//	- Do not change anything here
//================================================================================
int main(int argc, char* argv[])
{
    std::cout << "CV/task2 framework version 1.0"
              << std::endl;  // DO NOT REMOVE THIS LINE!!!
    std::cout << "===================================" << std::endl;
    std::cout << "               CV Task 2           " << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << CV_VERSION << std::endl;

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    std::ifstream fs(argv[1]);
    if (!fs)
    {
        std::cout << "Error: Failed to open file " << argv[1] << std::endl;
        return 2;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();

    rapidjson::Document doc;
    rapidjson::ParseResult check;
    check = doc.Parse<0>(buffer.str().c_str());

    if (check)
    {
        if (doc.HasMember("testcases"))
        {
            rapidjson::Value& testcases = doc["testcases"];
            for (rapidjson::SizeType i = 0; i < testcases.Size(); i++)
            {
                rapidjson::Value& testcase = testcases[i];
                executeTestcase(testcase, false);
#if BONUS
                std::cout << "\n\n\n\nStarting BONUS Task..." << std::endl;
                executeTestcase(testcase, true);
#endif
            }
        }
        cout << "\033[1;32m Program exited normally!\n \033[0m" << endl;
    }
    else
    {
        std::cout << "Error: Failed to parse file " << argv[1] << ":"
                  << check.Offset() << std::endl;
        return 3;
    }

    return 0;
}


void saveOutput(const Mat &output, const string &name, const string &out_path, bool bonus)
{
    // let's use png (?)
    string out_name = name + ".png";
    if(bonus)
        out_name = "bonus/" + out_name;
    else
        out_name = "normal/" + out_name;
    
    Mat out;
    if(output.type() == CV_32FC1 || output.type() == CV_32SC1)
    {
        double minVal, maxVal;
        minMaxLoc(output, &minVal, &maxVal);
        output.convertTo(out, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
    }
    else
    {
        out = output;
    }
    
    imwrite(out_path + out_name, out);
}




void printBinboostProperties(const BinBoostProperties& properties)
{
    cout << "----------------------------------------------" << endl;
    cout << "Properties of the binboost feature descriptor:" << endl;
    cout << "patchSize:   " << properties.patchSize << endl;
    //cout << "binLookUp:   " << properties.binLookUp << endl;
    cout << "gradAssign:  " << (properties.gradAssignType==ASSIGN_SOFT ? "ASSIGN_SOFT" : "other") << endl;
    cout << "nDim:        " << properties.nDim << endl;
    cout << "nWLs:        " << properties.nWLs << endl;
    cout << "orientQuant: " << properties.orientQuant << endl;
    cout << "----------------------------------------------" << endl;
}

void getPixelsOnCircle(int radius, std::vector<Point>& points)
{
    float fraction = M_PI / 56;
    
    int lastX = 0;
    int lastY = 0;
    for (int i = 0; i < 2 * 56; i++)
    {
        int x = round(radius * cos(i * fraction));
        int y = round(radius * sin(i * fraction));
        if (lastX != x || lastY != y)
        {
            lastX = x;
            lastY = y;
            Point p(x, y);
            points.push_back(p);
        }
    }
}


//==============================================================================
// makeScaleSpaceImage(const std::vector<std::vector<cv::Mat> > &scaleSpace, 
//                     cv::Mat &result)
//------------------------------------------------------------------------------
// creates a single image from the vector of vectors. for debugging purposes only
//
// Parameters:
// const std::vector<std::vector<cv::Mat> >& scaleSpace: The scale space
// cv::Mat& result: The result image
//==============================================================================
void makeScaleSpaceImage(const std::vector<std::vector<cv::Mat> > &scaleSpace, cv::Mat &result)
{
    result = cv::Mat::zeros(1, 1, CV_8UC1);

    if (scaleSpace.empty())
        return;

    // some sanity checks...

    const int numOctavesToDump = 3;
    const int start_idx = std::max(0, static_cast<int>(scaleSpace.size()) - numOctavesToDump);
    const int stop_idx = static_cast<int>(scaleSpace.size());
    if (start_idx == stop_idx)
        return;
    int image_width = 0;
    int image_height = 0;
    int num_layers = -1;
    for (int i = start_idx; i < stop_idx; i++)
    {
        if (scaleSpace[i].empty())
            return;

        int octave_width = scaleSpace[i][0].cols;
        int octave_height = scaleSpace[i][0].rows;
        if (num_layers == -1)
            num_layers = scaleSpace[i].size();

        if (scaleSpace[i].size() != num_layers)
        {
            cerr << "Number of layers in scale space does not match up" << std::endl;
            return;
        }
        // sanity check
        for (size_t j = 1; j < scaleSpace[i].size(); j++)
        {
            if (scaleSpace[i][j].rows != octave_height ||
                scaleSpace[i][j].cols != octave_width)
            {
                // bail out
                cerr << "Scale space dimensions do not match up" << std::endl;
                return;
            }
        }

        image_width += octave_width;
        image_height = std::max(image_height, octave_height);
    }

    result = cv::Mat::zeros(image_height * num_layers, image_width, CV_8UC1);

    int x = 0;
    for (int i = start_idx; i < stop_idx; i++)
    {
        int y = 0;
        int octave_width = scaleSpace[i][0].cols;
        int octave_height = scaleSpace[i][0].rows;
        for (int j = 0; j < scaleSpace[i].size(); j++)
        {
            scaleSpace[i][j].copyTo(result(cv::Rect(x, y, octave_width, octave_height)));
            y += octave_height;
        }
        x += octave_width;
    }
}


//==============================================================================
// executeTestcase(rapidjson::Value& testcase, bool bonus)
//------------------------------------------------------------------------------
// Executes the testcase.
//
// Parameters:
// rapidjson::Value& testcase: The json data of the testcase.
//==============================================================================
void executeTestcase(rapidjson::Value& testcase, bool bonus)
{
    cout << "ExecuteTestcase()" << endl;
    theRNG().state = hash<string>()("CV2 S2018");
    string data_folder = "data/";
    string name = testcase["name"].GetString();
    string testcase_folder = testcase["folder"].GetString();
    string input_left_name = testcase["input_left"].GetString();
    string input_right_name = testcase["input_right"].GetString();
    float match_distance_factor = float(testcase["match_distance_factor"].GetDouble());
    float diff_threshold = float(testcase["diff_threshold"].GetDouble());
    float diff_sigma = float(testcase["diff_sigma"].GetDouble());

    cout << "Folder: " << testcase_folder << endl;
    BinBoostProperties binBoostProperties;
    if(testcase.HasMember((const char*)"wl_bin_name"))
        binBoostProperties.wl_bin_path = data_folder + "/" + testcase["wl_bin_name"].GetString();
    else
        binBoostProperties.wl_bin_path = data_folder + "/binboost_128.bin";
    
    vector<Point> circle;
    int circle_radius = 3;
    getPixelsOnCircle(circle_radius, circle);
    string output = "./output/";


#if defined(_WIN32)
    _mkdir(output.c_str());
#else
    mkdir(output.c_str(), 0777);
#endif
    
    output = output.append(testcase_folder);

#if defined(_WIN32)
    _mkdir(output.c_str());
    _mkdir((output + "/normal").c_str());
    _mkdir((output + + "/bonus").c_str());
#else
    mkdir(output.c_str(), 0777);
    if(!testcase.HasMember((const char*)"only_bonus") || testcase["only_bonus"] != 1)
        mkdir((output + "/normal").c_str(), 0777);
#if BONUS
    mkdir((output + "/bonus").c_str(), 0777);
#endif /*BONUS */

#endif
    
    output += "/";

    cout << "Running testcase " << name << endl;

    START_TIMER(time_combined)
    string input_left_path = data_folder + "/" + input_left_name;
    string input_right_path = data_folder + "/" + input_right_name;

    cv::Mat image_left = cv::imread(input_left_path);
    cv::Mat image_right = cv::imread(input_right_path);
    
    cout << "read images" << endl;
    
    readWLsBinForBinboost(binBoostProperties);
    
    printBinboostProperties(binBoostProperties);
    
    cout << "loaded WeakLearner Binary" << endl;
    
    std::vector<KeyPoint> features_left, features_right;
    std::vector<KeyPoint> features_left_tmp, features_right_tmp;
    Mat descriptors_left, descriptors_right;
    Mat integral_image_left_out, Iyy_left_out, Ixx_left_out, Ixy_left_out, response_left_out;
    Mat integral_image_right_out, Iyy_right_out, Ixx_right_out, Ixy_right_out, response_right_out;
    Mat image_left_features_marked_out, image_right_features_marked_out, image_connected_features;
    Mat image_right_transformed; //##############If initialized here, the binBoost is not working!!! Some weird error
    Mat image_right_transformed_resized;
    Mat image_right_transformed_border;
    Mat image_transformed_diff;
    Mat image_transformed_diff_threshold;
    Mat img_matches, img_bad_matches;
    Mat image_contours;
    Mat image_contours_and_centers;
    Mat image_left_differences_marked_rect;

    vector<Mat> left_gradients, right_gradients;
    vector<Mat> left_integrals, right_integrals;
    Mat left_directions, right_directions;
    Mat left_magnitudes, right_magnitudes;
    
    vector<DMatch>      matches;
    std::vector<DMatch> good_matches, bad_matches;
    vector<Point2f>     good_matches_left, good_matches_right;

    Mat homography;
    vector<Point2f> corners_left(4);
    vector<Point2f> corners_right(4);

    vector<vector<Point>> contours;
    vector<Point2f> centers;
    vector<int> radius;
    vector<Rect> rect_diffs;
    vector<vector<Mat>> pyramid_left, pyramid_right, dog_pyramid_left, dog_pyramid_right;

    cout << "\n\n\033[1;32m----------------------------------------------------\n---- START TESTCASE ---- \033[0m" << endl;


    START_TIMER(time_features)
    if (bonus)
    {
        //-------------------------------------------------------------------------------------------------------------------------------------
        cout << "----------------------------------------------------\nBONUS: Calculate features+descriptors" << endl;
        binBoostFeaturesFast(image_left,  binBoostProperties,  left_gradients,  left_integrals,  left_directions,  left_magnitudes, features_left,  descriptors_left,  circle);
        binBoostFeaturesFast(image_right, binBoostProperties, right_gradients, right_integrals, right_directions, right_magnitudes, features_right, descriptors_right, circle);
    }
    else
    {
        if(!testcase.HasMember((const char*)"only_bonus") || testcase["only_bonus"] != 1)
        {
            //-------------------------------------------------------------------------------------------------------------------------------------
            cout << "----------------------------------------------------\nCalculate features+descriptors:" << endl;
            binBoostFeaturesHarrisLaplace(image_left, binBoostProperties, 
                                          pyramid_left, dog_pyramid_left,
                                          left_gradients, left_integrals,
                                          left_directions, left_magnitudes, features_left, descriptors_left);
            binBoostFeaturesHarrisLaplace(image_right, binBoostProperties, 
                                          pyramid_right, dog_pyramid_right,
                                          right_gradients, right_integrals,
                                          right_directions, right_magnitudes, features_right, descriptors_right);
        } else {
            //-------------------------------------------------------------------------------------------------------------------------------------
            cout << "----------------------------------------------------\n\033[1;32m!!!! THIS EXAMPLE IS ONLY FOR BONUS IMPLEMENTATION !!!!\033[0m" << endl;
            return;
        }
    }

    // we dump (part of) the scale space pyramid (but just for the left image for simplicity reasons...). We dump the last two octaves.
    if (!pyramid_left.empty())
    {
        cv::Mat gauss_debug_left;
        makeScaleSpaceImage(pyramid_left, gauss_debug_left);

        saveOutput(gauss_debug_left, "image_pyramid_left", output, bonus);
    }

    if (!dog_pyramid_left.empty())
    {
        cv::Mat dog_debug_left;
        makeScaleSpaceImage(dog_pyramid_left, dog_debug_left);
        cv::normalize(dog_debug_left, dog_debug_left, 0, 255, NORM_MINMAX, CV_8UC1);

        saveOutput(dog_debug_left, "dog_pyramid_left", output, bonus);
    }

    // test the harris detector
    {
        cv::Mat R, gray;
        cv::Mat localMax;
        cv::cvtColor(image_left, gray, cv::COLOR_BGR2GRAY);

        calculateHarrisCornerMeasure(gray, 1, 1, R);
        nonMaximaSuppressionHarris(R, localMax);

        cv::normalize(R, R, 0, 255, NORM_MINMAX, CV_8UC1);
        cv::normalize(localMax, localMax, 0, 255, NORM_MINMAX, CV_8UC1);

        saveOutput(R, "harris_left", output, bonus);
        saveOutput(localMax, "harris_nms_left", output, bonus);
    }

    // test rectifyPatch and the descriptor in isolation
    {
        cv::Mat patch = cv::Mat::zeros(10,10,CV_8UC1), gray;
        cv::cvtColor(image_left, gray, cv::COLOR_BGR2GRAY);
        int x = image_left.cols / 2;
        int y = image_left.rows / 2;
        cv::KeyPoint kp(x, y, 60, 90, 42.f, 0); // first layer, first octave, ...

        rectifyPatch(gray, kp, 32, patch);
        if (patch.empty())
            patch = cv::Mat::zeros(10,10,CV_8UC1);

        saveOutput(patch, "rectifyPatch", output, bonus);

        // also test the descriptor on this rectified patch...
        cv::Mat descriptor(1, ceil(binBoostProperties.nDim/8), CV_8UC1, cv::Scalar::all(0));
        std::vector<cv::KeyPoint> features;
        features.push_back(kp);
        computeBinboostDescriptorsRectify(gray, binBoostProperties, features, descriptor);

        saveOutput(descriptor, "descriptor_patch_center_left", output, bonus);
    }
    
    if(features_left.size() == 0 || features_right.size() == 0)
        cout << "\033[1;31m!!!! No features detected. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        cout << "\033[1;32m ##Features:\tleft: " << features_left.size()    << "\tright:" << features_right.size() << "\033[0m" << endl;

    if(features_left.size() == 0 || features_right.size() == 0)
        cout << "\033[1;31m!!!! No descriptors detected. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        cout << "\033[1;32m ##Descriptors:\tleft:" << descriptors_left.size() << "\tright:" << descriptors_right.size() << "\033[0m" << endl;

    if(left_directions.rows > 0)
        saveOutput( left_directions, "gradient_directions_left", output, bonus);
    if(right_directions.rows > 0)
        saveOutput(right_directions, "gradient_directions_right", output, bonus);
    if(left_magnitudes.rows > 0)
        saveOutput( left_magnitudes, "gradient_magnitudes_left", output, bonus);
    if(right_magnitudes.rows > 0)
        saveOutput(right_magnitudes, "gradient_magnitudes_right", output, bonus);

    drawKeypoints( image_left, features_left, image_left_features_marked_out, Scalar(0,0,255) );
    drawKeypoints( image_right, features_right, image_right_features_marked_out, Scalar(0,0,255) );
    saveOutput(image_left_features_marked_out, "features_marked_left", output, bonus);
    saveOutput(image_right_features_marked_out, "features_marked_right", output, bonus);

    //clone and set size to print in bonus
    features_left_tmp = features_left;
    features_right_tmp = features_right;
    for(KeyPoint& kp: features_left_tmp)
        if(kp.size == 0)
            kp.size = 15;
    for(KeyPoint& kp: features_right_tmp)
        if(kp.size == 0)
            kp.size = 15;
    drawKeypoints( image_left, features_left_tmp, image_left_features_marked_out, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    drawKeypoints( image_right, features_right_tmp, image_right_features_marked_out, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    saveOutput(image_left_features_marked_out, "features_rich_marked_left", output, bonus);
    saveOutput(image_right_features_marked_out, "features_rich_marked_right", output, bonus);
    features_left_tmp.clear();
    features_right_tmp.clear();

    for(int k = 0; k <= binBoostProperties.orientQuant && k < left_integrals.size() && k < right_integrals.size(); k++)
    {
        saveOutput(left_integrals[k], "integral_left_" + to_string(k), output, bonus);
        saveOutput(right_integrals[k], "integral_right_" + to_string(k), output, bonus);
        if(k < binBoostProperties.orientQuant)
        {
            saveOutput(left_gradients[k], "gradient_left_" + to_string(k), output, bonus);
            saveOutput(right_gradients[k], "gradient_right_" + to_string(k), output, bonus);
        }

    }

    if(descriptors_left.rows > 0)
        saveOutput(descriptors_left, "descriptors_left", output, bonus);
    if(descriptors_left.cols > 0)
        saveOutput(descriptors_right, "descriptors_right", output, bonus);





    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nCalculate matches:" << endl;
    cvflann::seed_random(42);
    BFMatcher matcher(NORM_HAMMING);
    matchFeatures(features_left, descriptors_left, features_right, descriptors_right, match_distance_factor, matches, good_matches_left, good_matches_right, matcher);
    
    if(matches.size() <= 2)
        cout << "\033[1;31m!!!! You have found <=2 matches. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        cout << "\033[1;32m ##Matches: " << matches.size() << "\033[0m" << endl;



    
    
    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nCalculate Homography" << endl;
    //-- Localize the object
    homography = calculateHomography(good_matches_left, good_matches_right);


    vector<bool> inliers;
    computeInliers(matches, homography, good_matches_left, good_matches_right, inliers);
    for (size_t i = 0; i < inliers.size(); i++)
    {
        if (inliers[i]) 
            good_matches.push_back(matches[i]);
        else
            bad_matches.push_back(matches[i]);
    }
    drawMatches(image_left, features_left, image_right, features_right, good_matches, img_matches,
                Scalar(0,0,255), Scalar(0,255,0), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(image_left, features_left, image_right, features_right, bad_matches, img_bad_matches,
                Scalar(0,0,255), Scalar(0,255,0), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    saveOutput(img_matches, "matches_good", output, bonus);
    saveOutput(img_bad_matches, "matches_bad", output, bonus);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    corners_left[0] = cvPoint(0,0);
    corners_left[1] = cvPoint( image_left.cols, 0 );
    corners_left[2] = cvPoint( image_left.cols, image_left.rows );
    corners_left[3] = cvPoint( 0, image_left.rows );

    perspectiveTransform(corners_left, corners_right, homography);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, corners_right[0] + Point2f( image_left.cols, 0), corners_right[1] + Point2f( image_left.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, corners_right[1] + Point2f( image_left.cols, 0), corners_right[2] + Point2f( image_left.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, corners_right[2] + Point2f( image_left.cols, 0), corners_right[3] + Point2f( image_left.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, corners_right[3] + Point2f( image_left.cols, 0), corners_right[0] + Point2f( image_left.cols, 0), Scalar( 0, 255, 0), 4 );


    //-- Draw corner-names
    //putText(img_matches, "top_left",     corners_right[0] + Point2f( image_left.cols      ,  10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
    //putText(img_matches, "top_right",    corners_right[1] + Point2f( image_left.cols -  70,  10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
    //putText(img_matches, "bottom_right", corners_right[2] + Point2f( image_left.cols - 100, -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);
    //putText(img_matches, "bottom_left",  corners_right[3] + Point2f( image_left.cols      , -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 2);

    //-- Show detected matches
    saveOutput(img_matches, "homography", output, bonus);



    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nTransform and resize image" << endl;

    transformImage(image_right, image_left, homography, image_right_transformed_resized);
    //transformImage(image_right, image_left, corners_right, image_right_transformed, image_right_transformed_resized);
    //if(image_right_transformed.rows == 0)
    //    cout << "\033[1;31m!!!! Transformation image empty. Check the previous steps !!!!\033[0m\n" <<  endl;
    //else
    //    saveOutput(image_right_transformed, "transformed_right", output, bonus);

    if(image_right_transformed_resized.rows == 0)
        cout << "\033[1;31m!!!! Resized image empty. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        saveOutput(image_right_transformed_resized, "transformed_right_resized", output, bonus);

    saveOutput(image_left, "transformed_right_resized_comparison_original_left", output, bonus);

    //warpPerspective(img, transformedImg, perspective_transform, Size(width, height), INTER_LINEAR);
    //cv::warpPerspective(image_right, transformedImg2, homography, Size(image_right.cols, image_right.rows), INTER_LINEAR+CV_WARP_INVERSE_MAP);





    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nCalculate difference and threashold image:" << endl;

    computeDiffs(image_left, image_right_transformed_resized, diff_sigma, diff_threshold, image_transformed_diff, image_transformed_diff_threshold, contours);
    if(image_right_transformed_resized.rows == 0)
        cout << "\033[1;31m!!!! Difference image empty. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        saveOutput(image_transformed_diff, "transformed_diff", output, bonus);

    if(image_right_transformed_resized.rows == 0)
        cout << "\033[1;31m!!!! Treholded image empty. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        saveOutput(image_transformed_diff_threshold, "transformed_diff_threshold", output, bonus);

    if(contours.size() == 0)
        if (testcase.HasMember((const char *) "no_diff") && testcase["no_diff"] == 1)
            cout << "\033[1;33m!!!! No contours found. THERE are no diffs in this example!!!!\033[0m\n" << endl;
        else
            cout << "\033[1;31m!!!! No contours found. Check the previous steps !!!!\033[0m\n" << endl;
    else
        cout << "\033[1;32m ##Contours:  " << contours.size() << "\033[0m" << endl;




    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nCalculate contour points: " << endl;

    // Max size of a difference is 1/3 of the image
    markDiffPoints(contours, image_left.cols/3, centers, radius, rect_diffs);

    if(centers.size() == 0 || radius.size() == 0 || rect_diffs.size() == 0)
        if (testcase.HasMember((const char *) "no_diff") && testcase["no_diff"] == 1)
            cout << "\033[1;33m!!!! Contour points not calculated. THERE are no diffs in this example!!!!\033[0m\n" << endl;
        else
            cout << "\033[1;31m!!!! Contour points not calculated. Check the previous steps !!!!\033[0m\n" <<  endl;
    else
        cout << "\033[1;32m ##Centers:  " << centers.size() << " Radius: " << radius.size() << " Rectangles: " << rect_diffs.size() << "\033[0m" << endl;





    //-------------------------------------------------------------------------------------------------------------------------------------
    cout << "\n----------------------------------------------------\nDraw differences" << endl;

    RNG rng(12345);
    // draw contours
    image_contours_and_centers = Mat::zeros(image_left.size(), CV_8UC3);
    image_contours             = Mat::zeros(image_left.size(), CV_8UC3);
    for( int i = 0; i < contours.size(); i++ )
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(image_contours_and_centers, contours, i, Scalar(255,255,255), 1, 8, noArray(), 0, Point());
        drawContours(image_contours            , contours, i, Scalar(255,255,255), 1, 8, noArray(), 0, Point());
        cv::circle(image_contours_and_centers, centers[i], radius[i], color, 2);
    }
    saveOutput(image_contours, "difference_contours", output, bonus);
    saveOutput(image_contours_and_centers, "difference_contours_and_circles", output, bonus);

    rng = RNG(12345);
    Mat image_left_differences_marked = image_left.clone();
    for( int i = 0; i < contours.size(); i++ )
    {
        Scalar c = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle  (image_left_differences_marked, centers[i], radius[i], c, 2);
    }
    saveOutput(image_left_differences_marked, "difference_contours_and_circles_in_image", output, bonus);


    image_left_differences_marked_rect = image_left.clone();
    for( auto rect : rect_diffs)
    {
        Scalar c = Scalar(0,0,255);//Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(image_left_differences_marked_rect, rect, c, 2);
    }
    saveOutput(image_left_differences_marked_rect, "difference_rect_marked_left_img", output, bonus);

    cout << "\n----------------------------------------------------" << endl;



    STOP_TIMER(time_features);

    cout << "\033[1;32m ---- Testcase finished normally    with: " << rect_diffs.size() << " differences found \n\033[0m" << endl;

    STOP_TIMER(time_combined)
}
