#include "algorithms.h"
#include <algorithm>
#define RAD2DEG (180.0 / CV_PI)

//================================================================================
// nonMaximaSuppression()
//--------------------------------------------------------------------------------
// TODO:
//  - depending on the gradient direction of the pixel classify each pixel P in one
//    of the following classes:
//    ____________________________________________________________________________
//    | class |direction                | corresponding pixels Q, R              |
//    |-------|-------------------------|----------------------------------------|
//    | I     | β <= 22.5 or β > 157.5  | Q: same row (y), left column (x−1)     |
//    |       |                         | R: same row (y), right column (x+1)    |
//    |-------|-------------------------|----------------------------------------|
//    | II    | 22.5 < β <= 67.5        | Q: row below (y+1), left column (x−1)  |
//    |       |                         | R: row above (y-1), left column (x−1)  |
//    |-------|-------------------------|----------------------------------------|
//    | III   | 67.5 < β <= 112.5       | Q: row above (y-1), same column (x)    |
//    |       |                         | R: row below (y+1), same column (x)    |
//    |-------|-------------------------|----------------------------------------|
//    | IV    | 112.5 < β <= 157.5      | Q: row below (y+1), left column (x−1)  |
//    |       |                         | R: row above (y-1), left column (x−1)  |
//    |_______|_________________________|________________________________________|
//  - compare the value of P with the values of Q and R:
//    if Q or R are greater than P -> set P to 0
//
// parameters:
//  - gradient_image: single channel 8-bit uchar matrix containing
//                    the combined gradienten image
//  - gradient_direction: single channel 32-bit float matrix containing
//                        the directions
// return:
//    single channel 32-bit float matrix containing the remaining lines
//================================================================================

Mat nonMaximaSuppression(const Mat &gradient_image, const Mat &gradient_directions) {
    Mat non_max_sup;
    gradient_image.copyTo(non_max_sup);
    gradient_image.convertTo(non_max_sup, CV_32F);

    float beta = 0.0;
    float q = 0.0;
    float r = 0.0;

    for (int rows = 1; rows < gradient_image.rows - 1; ++rows)
    {

        for (int cols = 1; cols < gradient_image.cols - 1; ++cols)
        {

            beta = gradient_directions.at<float>(rows,cols);
            if(gradient_directions.at<float>(rows,cols) < 0.f)
            {
                beta = gradient_directions.at<float>(rows,cols);
                beta = fmod(beta + 360.f, 180.f);
            }
            if( beta <= 22.5 || beta > 157.5 )
            {
                q = gradient_image.at<float>(rows,cols-1);
                r = gradient_image.at<float>(rows,cols+1);
            }
            else if( 22.5 < beta && beta <= 67.5 )
            {
                q = gradient_image.at<float>(rows+1,cols+1);
                r = gradient_image.at<float>(rows-1,cols-1);
            }
            else if( 67.5 < beta && beta <= 112.5 )
            {
                q = gradient_image.at<float>(rows-1,cols);
                r = gradient_image.at<float>(rows+1,cols);
            }
            else if( 112.5 < beta && beta <= 157.5)
            {
                q = gradient_image.at<float>(rows+1,cols-1);
                r = gradient_image.at<float>(rows-1,cols+1);
            }
            if ( gradient_image.at<float>(rows,cols) < q || gradient_image.at<float>(rows,cols) < r )
            {
                non_max_sup.at<float>(rows,cols) = 0.f;
            }
        }
    }
    return non_max_sup;
}

//================================================================================
// calcAngles()
//--------------------------------------------------------------------------------
// TODO:
//  - calculate for every pixel in the image the gradient direction
//    theta = tan(G_y/G_x)
// parameters:
//  - grad_x: single channel 32-bit float matrix containing
//            the gradient in x direction
//  - grad_y: single channel 32-bit float matrix containing
//            the gradient in y direction
// return:
//    a single channel 32-bit float matrix containing the calculated directions
//================================================================================
Mat calcAngles(const Mat &grad_x, const Mat &grad_y) {
    Mat directions(grad_x.size(), CV_32F, Scalar(0));

    for (int i = 0; i < grad_x.rows ; ++i)
    {
        for (int j = 0; j < grad_x.cols; ++j)
        {
            directions.at<float>(i,j) = (float)atan2(grad_y.at<float>(i,j),grad_x.at<float>(i,j))*RAD2DEG;
        }
    }


    return directions;
}



void recursion(Mat thresh,  int x, int y, uchar threshold_min, uchar threshold_max)
{

    for (int i = x - 1; i < x + 2; i++)
    {
        for ( int j = y - 1; j < y + 2; j++)
        {
            if (i < 0 || i > thresh.rows || j < 0 || j > thresh.cols) continue;
            if ((thresh.at<uchar>(i,j) < 255) && (thresh.at<uchar>(i,j) > 0))
            {
                thresh.at<uchar>(x, y) = 255;
                recursion(thresh,i,j,threshold_min,threshold_max);
            }
        }
    }
    return;
}


//================================================================================
// hysteresis()
//--------------------------------------------------------------------------------
// TODO:
//  - set all pixels under the lower threshold to 0
//  - set all pixels over the high threshold to 255
//  - classify all weak edges (thres_min <= weak edge < thres_max)
//    - if one of the the 8 surrounding pixel values is higher than thres_max,
//      also the weak pixel is a strong pixel
//    - check this recursively to be sure not to miss one
//  - set all remaining, not classified pixels to 0
// parameters:
//  - non_max_sup: single channel 32-bit float matrix containing the remaining lines
//  - thresh: single channel 8-bit uchar matrix holding the results of the hysteresisi calulation
//  - threshold_min: the lower threshold
//  - threshold_min: the upper threshold
//================================================================================
void hysteresis(const Mat &non_max_sup, Mat &thresh, uchar threshold_min, uchar threshold_max) {
    non_max_sup.convertTo(thresh, CV_8U);

    Point_<int> coord;
    std::vector <Point> strong;

    //saving all strong values

    for (int row = 0; row < thresh.rows; row ++)
    {
        for (int col = 0; col < thresh.cols; col ++)
        {
            if (thresh.at<uchar>(row, col) >= threshold_max)
            {
                thresh.at<uchar>(row, col) = 255;
                coord.x = row;
                coord.y = col;
                strong.push_back(coord);
            }
            if (thresh.at<uchar>(row, col) < threshold_min)
            {
                thresh.at<uchar>(row, col) = 0;

            }
        }
    }

    //checking all week values an seting them to 255...if possible.

    for (auto it = strong.begin() ; it != strong.end(); ++it)
    {
        recursion(thresh,it.base()->x,it.base()->y,threshold_min,threshold_max);
    }


    //setting rest to 0

    for (int row = 0; row < thresh.rows; row ++)
    {
        for (int col = 0; col < thresh.cols; col ++)
        {
            if ((thresh.at<uchar>(row,col) < 255) && (thresh.at<uchar>(row,col) > 0))
            {
                thresh.at<uchar>(row,col) = 0;
            }

        }
    }



}

//================================================================================
// cannyOwn()
//--------------------------------------------------------------------------------
// TODO:
//  - calculate the gradient images
//  - calculate the gradient directions (hint: calcAngles)
//  - caluclate the Non-Maxima-Suppression (hint: nonMaximaSuppression)
//  - calculate the hysteresis (hint: hysteresis)
// parameters:
//  - image: 8-bit singel channel matrix containign the blurred grayscale image
//  - end_result: the end result of the canny algorithm
//  - threshold_min: the lower threshold
//  - threshold_min: the upper threshold
//  - grad_x: the gradient image in x direction
//  - grad_y: the gradient image in y direction
//  - grad: the combined gradient image
//  - gradient_directions: the directions of the gradients
//  - non_maxima_suppression: the remaining lines after the non maxima suppression
//================================================================================
void algorithms::cannyOwn(const Mat &image, Mat &end_result, const uchar threshold_min, const uchar threshold_max,
                          Mat &grad_x, Mat &grad_y, Mat &grad, Mat &gradient_directions, Mat &non_maxima_suppression) {
    // TODO: Gradient computation (grad_x, grad_y, grad)
    //G x = S x ∗ I
    //G y = S y ∗ I
    Mat gradient_X_quadratic;
    Mat gradient_Y_quadratic;

    cv::Sobel(image, grad_x, CV_32F, 1,0,3);
    cv::Sobel(image, grad_y, CV_32F, 0,1,3);

    //G_x^2
    cv::multiply(grad_x,grad_x,gradient_X_quadratic);

    //G_y^2
    cv::multiply(grad_y,grad_y,gradient_Y_quadratic);

    //| G |= G_x^2 + G_y^2
    cv::sqrt(gradient_X_quadratic + gradient_Y_quadratic, grad);

    gradient_directions = calcAngles(grad_x, grad_y);
    non_maxima_suppression = nonMaximaSuppression(grad, gradient_directions);
    hysteresis(non_maxima_suppression, end_result, threshold_min, threshold_max);
}

struct sorterStruct{
    inline bool operator() (algorithms::CvLocalMaximum &a, algorithms::CvLocalMaximum &b)
    {
        return a.accumulator_value > b.accumulator_value;
    }
};

//================================================================================
// Hough transform for circles
// HoughCirclesOwn()
//--------------------------------------------------------------------------------
// TODO:
//  - fill out accumulator array
//  - find local maxima
//  - sort the local maxima
//  - store the first min(total, circlesMax) circles to the output buffer ``circles''
// Input:
//    img - 8-bit, single channel source image with non-zeros representing edges
//    rho_quant - distance resolution of the x-y coordinates of the circle
//    rad_quant - distance resolution of the radius of the circle
//    rad_min - minimum radius
//    rad_max - maximum radius
//    threshold - accumulator threshold parameter
//    circlesMax - maximum number of lines returned
// Returns:
//    circles - the detected circles, therefore a vector filled with (x, y, r)
//              3-tuples representing the x, y coordinates of a circle with
//              radius r
//    accum - vector of 2D matrices for the accumulator array. Each entry in
//            this vector represents an accumulator array for the x,y positions
//            of the circles
//    local_maximums - list of local maxima in the accumulator array
//================================================================================
void algorithms::HoughCirclesOwn(
        const Mat &img,
        vector<Vec3f> &circles,
        float spat_quant,
        float rad_quant,
        int rad_min,
        int rad_max,
        int threshold,
        int circlesMax,
        vector<Mat> &accum,
        vector<CvLocalMaximum> &local_maximums) {
    // cleanup vectors
    local_maximums.clear();
    accum.clear();
    circles.clear();

    // initial values
    int spatial_w = static_cast<int>(ceil(img.cols / spat_quant));
    int spatial_h = static_cast<int>(ceil(img.rows / spat_quant));
    auto num_rad = static_cast<size_t>(ceil((rad_max - rad_min) / rad_quant));
    accum.resize(num_rad);

    for (Mat &a : accum)
        a = Mat::zeros(spatial_h, spatial_w, CV_32SC1);

    // TODO: 1. fill out accumulator array




    for (int r = 0; r < num_rad ; ++r) {
        //radius  pixel za sve proci

        float r_circle = rad_min + (r + 1) * rad_quant;

        for (int x = 0; x < img.rows; ++x) {
            for (int y = 0; y < img.cols ; ++y) {

//kandidat razlicit od nule u boji
                if(!img.at<uchar>(x,y)) continue;

//koordinate 8nb zavisno od precnika
                int x_start = cvFloor((x - r_circle)/spat_quant - 1);
                int y_start = cvFloor((y - r_circle)/spat_quant - 1);
                int x_stop = cvCeil((x + r_circle)/spat_quant + 1);
                int y_stop = cvCeil((y + r_circle)/spat_quant + 1);

                for (int xroi = x_start; xroi < x_stop; ++xroi) {
                    for (int yroi = y_start; yroi < y_stop; ++yroi) {

//formule za racunanje provje provjerava piksele i region of interest
// njihove medju odnose. tamo gdje se najvise sijeku tu je centar kruga ipotrencialni novcic.
// U akumulatoru inkrementujes dijelove gdje se najvise sijeklo

                        float ycoord = (yroi + 0.5) * spat_quant;
                        float xcoord = (xroi + 0.5) * spat_quant;
                        float dy = ycoord - y;
                        float dx = xcoord - x;

                        float dcircle_sqrt = sqrt(dy*dy + dx*dx);
                        int dcircle = cvRound((dcircle_sqrt - rad_min)/rad_quant);

                        if(xroi < spatial_h && yroi < spatial_w && xroi >= 0 && yroi >= 0 && dcircle == r)
                        {
                            accum[r].at<int>(xroi,yroi) += 1;
                        }
                    }
                }
            }
        }
    }

    // TODO: 2. find local maxima


// za sve vrijednosti vece od thrasholda gledam 3d nb od tog pixela i ukoliko je pixel veci on je krug
    for (int r = 0; r < accum.size() ; ++r) {
        for (int x = 0; x < accum.at(r).rows ; ++x) {
            for (int y = 0; y < accum.at(r).cols ; ++y) {

                if(accum.at(r).at<int>(x,y) > threshold)
                {
                    bool flagic = false;
                    //check local 3D neighborhood
                    for (int dr = 0; dr < 3; ++dr) {
                        if (flagic) break;
                        for (int dx = 0; dx < 3; ++dx) {
                            if (flagic) break;
                            for (int dy = 0; dy < 3; ++dy) {

                                if(r + dr - 1 < 0 || x + dx - 1 < 0 ||  y + dy - 1 < 0 ||
                                   r + dr - 1 >= accum.size() || x + dx - 1 >= spatial_h ||  y + dy - 1 >= spatial_w)
                                {
                                    continue;
                                }

                                if(accum.at(r + dr - 1).at<int>(x+dx-1,y+dy-1) > accum.at(r).at<int>(x,y))
                                {
                                    flagic = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!flagic)
                    {
                        CvLocalMaximum localMaximum = {r,y,x,accum.at(r).at<int>(x,y)};
                        local_maximums.push_back(localMaximum);
                    }
                }

            }
        }
    }

    // TODO: 3. sort the local maxima

    std::sort(local_maximums.begin(),local_maximums.end(),sorterStruct());

    // TODO: 4. store the first min(total, circlesMax) circles to the output buffer ``circles''

    for (int iterator = 0; iterator < circlesMax; iterator++) {

        float centarx = (local_maximums.at(iterator).x + 0.5f) * spat_quant;
        float centary = (local_maximums.at(iterator).y + 0.5f) * spat_quant;
        float radius = (local_maximums.at(iterator).r + 0.5f) * rad_quant + rad_min;

        cv::Vec3f pusher = {centarx,centary,radius};
        circles.push_back(pusher);
    }

}



//================================================================================
// Hough transform for lines
// HoughLinesOwn()
//--------------------------------------------------------------------------------
// TODO: BONUS:
//  - precalculate needed sine and cosine values
//  - fill out accumulator array
//  - find local maxima
//  - sort the local maxima
//  - store the first min(total, linesMax) lines to the output buffer ``lines''
// Input:
//    img - 8-bit, single channel source image with non-zeros representing edges
//    rho - distance resolution of the accumulator in pixels
//    theta - angle resolution of the accumulator in radians. Means how fast the line in the Hough-space rotates
//    threshold - accumulator threshold parameter
//    linesMax - maximum number of lines returned
// Returns:
//    lines - the detected lines, therefore a vector filled with (rho, theta) pairs corresponding to a line
//    local_maximums - list of local maxima in the accumulator array
//================================================================================
void algorithms::HoughLinesOwn(Mat img, vector<Vec2f> &lines, float rho, float theta, int threshold, int linesMax,
                               Mat &accum, vector<CvLocalMaximum> &local_maximums) {
    int width = img.size().width;
    int height = img.size().height;
    int num_angle = cvRound(CV_PI / theta); // number of angles we have to test per point in the image
    int num_rho = cvRound(((width + height) * 2 + 1) /
                          rho); // equals the normalized diagonal of the image. Normalized by using the distance resolution of the accumulator
    int r, n; // indexes used in conjunction with num_angle and num_rho
    float tab_sin[num_angle], tab_cos[num_angle];

    // cleanup vectors/matrices
    lines.clear();
    local_maximums.clear();
    accum = Mat::zeros(Size(num_angle + 2, num_rho + 2),
                       CV_32SC1); // matrix of type INT, +2 to rows/cols to be able to compare the neighborhood using a cross-window




    // TODO: BONUS: 1. precalculate needed sine and cosine values

    // TODO: BONUS: 2. fill accumulator

    // TODO: BONUS: 3. find local maxima

    // TODO: BONUS: 4. sort the local maxima

    // TODO: BONUS: 5. store the first min(total, linesMax) lines to the output buffer
}

//================================================================================
// classifyCircles()
//--------------------------------------------------------------------------------
// TODO BONUS: classify each found circle and calculate the over all sum
// Input:
//  - circles: a vector of detected circles
//  - lines: a vector of detected lines
//  - coin_properties: the properties of possible coins
// Returns:
//    The counted value of all coins in the image
double algorithms::classifyCircles(const vector<Vec3f> &circles, const vector<Vec2f> &lines,
                                   const vector<algorithms::coin> coin_properties) {
    // BONUS: Implement this
    return -1.0;
}

void algorithms::Erosion(int erosion_type, int erosion_size, InputArray src, OutputArray dst) {
    // possible erosion types:
    //  MORPH_RECT = 0
    //  MORPH_CROSS = 1
    //  MORPH_ELLIPSE = 2
    Mat element = getStructuringElement(erosion_type,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    // apply the erosion operation
    erode(src, dst, element);

}

void algorithms::Dilation(int dilation_type, int dilation_size, InputArray src, OutputArray dst) {
    // possible dilation types:
    //  MORPH_RECT = 0
    //  MORPH_CROSS = 1
    //  MORPH_ELLIPSE = 2
    Mat element = getStructuringElement(dilation_type,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                        Point(dilation_size, dilation_size));
    // apply the dilation operation
    dilate(src, dst, element);
}


void algorithms::drawHoughLines(Mat &image, vector<Vec2f> lines) {
    for (const Vec2f &l : lines) {
        float rho = l[0], theta = l[1];
        Point pt1, pt2;
        float a = cos(theta), b = sin(theta);
        float x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(image, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
    }
}

void algorithms::drawHoughCircles(Mat &result, const vector<Vec3f> &circles) {
    for (const Vec3f &c : circles) {
        circle(result, Point2f(c[0], c[1]), static_cast<int>(c[2]), Scalar(0, 255, 0), 5);
    }
}


void algorithms::writeHoughAccum(const vector<Mat> &circleAccum, const string &fname) {
    if (circleAccum.empty()) {
        imwrite(fname, Mat::zeros(10, 10, CV_8UC1));
        return;
    }

    int rows = static_cast<int>(circleAccum.size()) * max(1, circleAccum[0].rows);
    int cols = max(1, circleAccum[0].cols);

    Mat result = Mat::zeros(rows, cols, CV_32SC1);
    for (int i = 0; i < circleAccum.size(); i++) {
        const Mat &accumArray = circleAccum[i];
        accumArray.copyTo(result.rowRange(i * circleAccum[0].rows, (i + 1) * circleAccum[0].rows));
    }

    normalize(result, result, 0, 255, NORM_MINMAX, CV_8UC1);

    imwrite(fname, result);
}
