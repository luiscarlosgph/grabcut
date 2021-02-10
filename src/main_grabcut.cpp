/**
 * @brief  Script to run a GMM based on a trimap or a rect.
 *
 * @date   20 August 2019.
 * @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
 */

#define STRCASECMP strcasecmp
#define STRNCASECMP strncasecmp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

// CUDA
#include <cuda_runtime.h>
#include <npp.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// My includes
#include <cmdlinereader.h>
#include <grabcut.h>

int main(int argc, char **argv) {
  // Parse command line
  if (!CommandLineReader::getInstance().processCmdLineOptions(argc, argv)) return EXIT_FAILURE;

  // Load image
  const std::string defaultImage = CommandLineReader::getInstance().getInputFilePath();
  cv::Mat image = cv::imread(defaultImage);
  if (!image.data) {
    std::cout << "Could not open or find the image" << std::endl;
    return EXIT_FAILURE;
  }

  // Convert image to BGRA
  cv::Mat bgra;
  cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
#ifndef NDEBUG
  cv::imshow("Original image", bgra);
  cv::waitKey();
#endif

  // Get maximum number of iterations from command line
  int maxIter = CommandLineReader::getInstance().getNumberOfIterations();

  // Load rect image and find rect coordinates
  cv::Mat result;
  cv::Mat probaResult;
  if (!CommandLineReader::getInstance().getRectFilePath().empty()) {
    // The user inputs an image with black background and a white rectangle on
    // the ROI

    cv::Mat rectColor = cv::imread(CommandLineReader::getInstance().getRectFilePath());
    cv::Mat rectGray;
    cv::cvtColor(rectColor, rectGray, cv::COLOR_BGR2GRAY);
#ifndef NDEBUG
    cv::imshow("Rect image loaded from file", rectGray);
    cv::waitKey();
#endif

    // Find rectangle coordinates and convert them to OpenCV rect
    cv::Rect rect;
    rect.y = -1;
    for (int i = 0; i < rectGray.rows; ++i) {
      uchar *p = rectGray.ptr<uchar>(i);
      for (int j = 0; j < rectGray.cols; ++j) {
        if (p[j] > 0 && rect.y == -1) {
          rect.y = i;
          rect.x = j;
        }
        if (p[j] > 0) {
          rect.height = i + 1 - rect.y;
          rect.width = j + 1 - rect.x;
        }
      }
    }

    // Grabcut segmentation
    GrabCut grabcut(maxIter);
    result = grabcut.estimateSegmentationFromRect(bgra, rect);
  } else if (!CommandLineReader::getInstance().getTrimapFilePath().empty()) {
    // We expect the input image in this format:
    //
    //    0   = sure background
    //    128 = unknown
    //    255 = sure foreground
    //
    cv::Mat trimap = 
      cv::imread(CommandLineReader::getInstance().getTrimapFilePath(), cv::IMREAD_GRAYSCALE);

    // Grabcut segmentation
    GrabCut grabcut(maxIter);
    result = grabcut.estimateSegmentationFromTrimap(bgra, trimap);
  } else if (!CommandLineReader::getInstance().getFourmapFilePath().empty()) {
    // We expect the input image in this format:
    //
    //    0   = sure background
    //    64  = probably background
    //    128 = probably foreground
    //    255 = sure foreground
    //
    cv::Mat fourmap =
      cv::imread(CommandLineReader::getInstance().getFourmapFilePath(), cv::IMREAD_GRAYSCALE);

    // Grabcut segmentation
    GrabCut grabcut(maxIter);
    result = grabcut.estimateSegmentationFromFourmap(bgra, fourmap); 
  } else if (!CommandLineReader::getInstance().getFgMapFilePath().empty()) {
    // Load background probability map
    cv::Mat bgMap =
      cv::imread(CommandLineReader::getInstance().getBgMapFilePath(), cv::IMREAD_GRAYSCALE);
    // Load foreground probability map
    cv::Mat fgMap =
      cv::imread(CommandLineReader::getInstance().getFgMapFilePath(), cv::IMREAD_GRAYSCALE);

    // Convert probability maps to [0, 1] range 
    cv::Mat probaBg, probaFg;
    bgMap.convertTo(probaBg, CV_32FC1);
    fgMap.convertTo(probaFg, CV_32FC1);
    probaBg /= 255.0f;
    probaFg /= 255.0f;

    // Perform cut using the probability maps to obtain the unary terms
    GrabCut grabcut(maxIter);
    double gamma = CommandLineReader::getInstance().getGamma();
    if (gamma != -1) {
      result = grabcut.estimateSegmentationFromProba(bgra, probaBg, probaFg, gamma);
    } 
    else {
      result = grabcut.estimateSegmentationFromProba(bgra, probaBg, probaFg);
    }
  } else {
    throw std::invalid_argument("Must provide either a rect, trimap or fourmap.");
  }

  // Save result in output file
  result *= 255;
  cv::imwrite(CommandLineReader::getInstance().getOutputFilePath(), result);
  
  return EXIT_SUCCESS;
}
