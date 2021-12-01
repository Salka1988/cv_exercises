#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "algorithms.h"

#define FULL_VERSION 1

#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

using namespace std;
using namespace cv;

int loadConfig(rapidjson::Document &config, const char *config_path) {
  FILE *fp = fopen(config_path, "r");
  if (!fp) {
    cout << BOLD(FRED("[ERROR]")) << " Reading File " << config_path << " failed\n" << endl;
    return -1;
  }
  char readBuffer[65536];
  rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  config.ParseStream(is);
  assert(config.IsObject());
  return 0;
}

void createDir(const char *path) {
#if defined(_WIN32)
  _mkdir(path);
#else
  mkdir(path, 0777);
#endif
}

vector<algorithms::coin> getCoinProperties(rapidjson::Document &config, const string coin_properties) {
  vector<algorithms::coin> coins;
  const rapidjson::Value &a = config[coin_properties.c_str()];
  for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
    algorithms::coin coin;
    coin.diameter = a[i]["diameter"].GetDouble();
    coin.value = a[i]["value"].GetDouble();
    coins.push_back(coin);
  }
  return coins;
}

vector<string> getDataSelections(rapidjson::Document &config, string data_selection) {
  const rapidjson::Value &a = config[data_selection.c_str()];
  vector<string> members;
  for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
    if (a[i].IsString()) {
      string data_selected = a[i].GetString();
      members.push_back(data_selected);
    }
  }
  return members;
}

vector<string>
getConfigFilenames(rapidjson::Document &config, int number_width, bool zero_filled, const string out_filename_array,
                   const string out_full_path, const string out_filetype) {
  const rapidjson::Value &a = config[out_filename_array.c_str()];
  vector<string> members;
  for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
    if (a[i].IsString()) {
      string out_filename = a[i].GetString();
      string digit_string = to_string(i + 1);
      string zero_digit_string = zero_filled ? string(number_width - digit_string.length(), '0') + digit_string
                                             : digit_string;
      string member = out_filename + zero_digit_string;
      members.emplace_back(out_full_path + zero_digit_string + "_" + out_filename + out_filetype);
//      cout << member << " = " << members.back() << endl;
    }
  }
  return members;
}

void run(Mat image, vector<string> out_filenames, vector<algorithms::coin> coin_properties, uchar threshold_min,
         uchar threshold_max, int radius_min, int radius_max, int threshold_hough_line, int threshold_hough_circle,
         double rho_hough_line, double rho_hough_circle, double theta_hough_line, double rad_hough_circle,
         int max_no_lines, int max_no_cirlces) {
  int out_filename_count = 0;
  Mat gray, gray_blur, thresh, kernel, closing, hierarchy, result, result_hough_test;


  // preparation of the image
  image.copyTo(result);
  image.copyTo(result_hough_test);
  cvtColor(image, gray, COLOR_BGR2GRAY);
  GaussianBlur(gray, gray_blur, Size(15, 15), 0);

  // canny edge detection
  Mat canny_grad_x, canny_grad_y, canny_grad, canny_angles, canny_non_maxima_suppression, canny_end_result, img_save;
  algorithms::cannyOwn(gray_blur, canny_end_result, threshold_min, threshold_max, canny_grad_x, canny_grad_y,
                       canny_grad, canny_angles,
                       canny_non_maxima_suppression);

  // save created images
  canny_grad_x.convertTo(img_save, CV_8UC3);
  imwrite(out_filenames[out_filename_count++], img_save);
  canny_grad_y.convertTo(img_save, CV_8UC3);
  imwrite(out_filenames[out_filename_count++], img_save);
  canny_grad.convertTo(img_save, CV_8UC3);
  imwrite(out_filenames[out_filename_count++], img_save);
  canny_angles.convertTo(img_save, CV_8UC3, 255.f / 360.f, 180);
  imwrite(out_filenames[out_filename_count++], img_save);
  imwrite(out_filenames[out_filename_count++], canny_non_maxima_suppression);
  imwrite(out_filenames[out_filename_count++], canny_end_result);
  Canny(gray_blur, thresh, threshold_min, threshold_max, 3, false);

  // improve result through dilation and erosion
  algorithms::Dilation(MORPH_ELLIPSE, 3, thresh, closing);
  // store result
  imwrite(out_filenames[out_filename_count++], closing);
  algorithms::Erosion(MORPH_ELLIPSE, 3, closing, closing);
  // store result
  imwrite(out_filenames[out_filename_count++], closing);

  vector<Vec3f> circles;
  vector<Mat> circleAccum;
  vector<algorithms::CvLocalMaximum> circleLocalMax;
  // Use the non-closed version here.
  algorithms::HoughCirclesOwn(thresh, circles, rho_hough_circle, rad_hough_circle,
                              radius_min, radius_max, threshold_hough_circle,
                              max_no_cirlces, circleAccum, circleLocalMax);
  algorithms::writeHoughAccum(circleAccum, out_filenames[out_filename_count++]);
  algorithms::drawHoughCircles(result, circles);

  // draw the coin contours
  //drawContours(result, closing);
  // store result
  imwrite(out_filenames[out_filename_count++], result);



  // own standard hough line transform
  Mat result_hough_test_own, accum;
  image.copyTo(result_hough_test_own);
  vector<Vec2f> lines_own, lines; // will hold the results of the hough line detection
  vector<algorithms::CvLocalMaximum> local_maximums; // CvLocalMaximum contains two int's, like a Vec2i
  // TODO: BONUS: Either use HoughLinesOwn or another method to find the rectangle.
  algorithms::HoughLinesOwn(closing, lines_own, (float) rho_hough_line, (float) theta_hough_line, threshold_hough_line,
                            max_no_lines, accum, local_maximums);

  // store accumulator image
  Mat accum_normalized_resized;
  normalize(accum, accum_normalized_resized, 0, 255, NORM_MINMAX,
            CV_8UC1); // normalize values between 0 and 255 (usually brightens the image)
  //  resize(accum_normalized_resized, accum_normalized_resized, Size(1000, 1000)); // resize image
  // store result
  imwrite(out_filenames[out_filename_count++], accum_normalized_resized);

  // write local maximums to an image -> image is just representing the matrix for comparing the values
  Mat mat_local_maximums = !local_maximums.empty() ? Mat(static_cast<int>(local_maximums.size()), 4, CV_32SC1, local_maximums.data()) : Mat::zeros(1, 1, CV_32SC1);
  imwrite(out_filenames[out_filename_count++], mat_local_maximums);
//  cout << "Linecount: " << lines.size() << endl;
  algorithms::drawHoughLines(result_hough_test_own, lines_own);
  // store result
  imwrite(out_filenames[out_filename_count++], result_hough_test_own);

  // standard hough line transform
  HoughLines(closing, lines, rho_hough_line, theta_hough_line, threshold_hough_line); // runs the actual detection
  algorithms::drawHoughLines(result_hough_test, lines);
  // store result
  imwrite(out_filenames[out_filename_count], result_hough_test);





  // TODO: BONUS: use the found rectangle here
  double value = algorithms::classifyCircles(circles, lines_own, coin_properties);
  if (value > 0) {
    cout << "BONUS IMPLEMENTED!" << endl;
    cout << "Aaaargh captain we found a treasure!" << endl;
    cout << "In the treasure chest are gold coins with a value of " << value << "." << endl;
    cout << "Let's buy a bottle of rum. Ahoy!" << endl;
  }
}


//==============================================================================
// main()
//
//==============================================================================
int main(int argc, char *argv[]) {
  printf(BOLD(FGRN("[INFO ]")));
  printf(" CV/task1b framework version 1.0\n");  // DO NOT REMOVE THIS LINE!

  bool load_default_config = argc == 1;
  bool load_argv_config = argc == 2;

  // check console arguments
  if (load_default_config) {
    cout << BOLD(FGRN("[INFO]")) << " No Testcase selected - using default Testcase (=0)\n" << endl;
  } else if (!load_argv_config) {
    cout << BOLD(FRED("[ERROR]")) << " Usage: ./cvtask1b <TC-NO. (0-3)>\n" << endl;
    return -1;
  }

  try {
    // load config
    rapidjson::Document config;
    int res = loadConfig(config, "config.json");
    if (res != 0)
      return -1;

    // input parameters
    vector<string> data_selections = getDataSelections(config, string("data_selected"));
    if (load_argv_config && atoi(argv[1]) >= data_selections.size()) {
      cout << BOLD(FRED("[ERROR]")) << " Comandline argument (= " << atoi(argv[1])
           << ") is higher than number of Testcases (= " << data_selections.size() - 1 << ")\n" << endl;
      return -1;
    }
    string data_selected = data_selections.at(load_default_config ? 0 : atoi(argv[1]));
    string data_path = config["data_path"].GetString();
    string out_directory = config["out_directory"].GetString();
    string out_full_path = out_directory + data_selected;
    string out_filetype = config["out_filetype"].GetString();

    int out_filename_number_width = config["out_filename_number_width"].GetInt();
    bool out_filename_number_zero_filled = config["out_filename_number_zero_filled"].GetBool();
    vector<string> out_filenames = getConfigFilenames(config, out_filename_number_width,
                                                      out_filename_number_zero_filled, string("out_filenames"),
                                                      out_full_path, out_filetype);
    vector<algorithms::coin> coin_properties = getCoinProperties(config, string("coin_properties"));

    // combine needed strings
    string data_full_path = data_path + data_selected;
    string data_config_full_path = data_full_path + "config.json";
    cout << BOLD(FGRN("[INFO]")) << " Data path: " << data_full_path << endl;
    cout << BOLD(FGRN("[INFO]")) << " Data config path: " << data_config_full_path << endl;
    cout << BOLD(FGRN("[INFO]")) << " Output path: " << out_full_path << endl;

    // load data config
    rapidjson::Document config_data;
    res = loadConfig(config_data, data_config_full_path.c_str());
    if (res != 0)
      return -1;

    // load data config content
    string image_path = config_data["image"].GetString();
    uchar threshold_min = static_cast<uchar>(config_data["theshold_canny_min"].GetInt());
    uchar threshold_max = static_cast<uchar>(config_data["theshold_canny_max"].GetInt());

    int radius_min = config_data["radius_circle_min"].GetInt();
    int radius_max = config_data["radius_circle_max"].GetInt();

    int threshold_hough_line = config_data["threshold_hough_line"].GetInt();
    int threshold_hough_circle = config_data["threshold_hough_circle"].GetInt();

    double rho_hough_line = config_data["rho_hough_line"].GetDouble();
    double rho_hough_circle = config_data["rho_hough_circle"].GetDouble();

    double theta_hough_line = config_data["theta_hough_line"].GetDouble();
    double rad_hough_circle = config_data["rad_hough_circle"].GetDouble();

    int max_no_lines = config_data["max_no_lines"].GetInt();
    int max_no_circles = config_data["max_no_circles"].GetInt();


    string image_full_path = data_full_path + image_path;
    cout << BOLD(FGRN("[INFO]")) << " Image path: " << image_full_path << endl;

    // create output dirs
    createDir(out_directory.c_str());
    createDir(out_full_path.c_str());

    // load input image
    Mat img_BGR = imread(image_full_path);
    // check if image was loaded
    assert(img_BGR.data);
    // display image

    run(img_BGR, out_filenames, coin_properties, threshold_min, threshold_max, radius_min, radius_max,
        threshold_hough_line, threshold_hough_circle, rho_hough_line, rho_hough_circle, theta_hough_line,
        rad_hough_circle, max_no_lines, max_no_circles);
  }
  catch (const exception &ex) {
    cout << ex.what() << endl;
    cout << BOLD(FRED("[ERROR]")) << " Program exited with errors!" << endl;
    return -1;
  }
  cout << BOLD(FGRN("[INFO ]")) << " Program exited normally!" << endl;
  return 0;
}
