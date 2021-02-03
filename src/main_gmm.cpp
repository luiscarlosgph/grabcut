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
#include <gmm.h>

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

  // Load rect image and find rect coordinates
  GMM *gmmPtr;
  std::chrono::duration<double> initElapsed;
  if (!CommandLineReader::getInstance().getRectFilePath().empty()) {
    // The user inputs an image with black background and a white rectangle on
    // the ROI

    cv::Mat rectColor = cv::imread(CommandLineReader::getInstance().getRectFilePath());
    cv::Mat rectGray;
    cv::cvtColor(rectColor, rectGray, CV_BGR2GRAY);
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

    // Create GMM
    // ptr = std::make_shared<int>(53);
    auto initStart = std::chrono::high_resolution_clock::now();
    gmmPtr = new GMM(GMM::Builder().withRect(bgra, rect).build());
    auto initFinish = std::chrono::high_resolution_clock::now();
    initElapsed = initFinish - initStart;
  } else {
    // The user inputs an image with black background and a trimap
    cv::Mat trimapColor = cv::imread(CommandLineReader::getInstance().getTrimapFilePath());
    cv::Mat trimapGray;
    cv::cvtColor(trimapColor, trimapGray, CV_BGR2GRAY);
#ifndef NDEBUG
    cv::imshow("Trimap loaded from file", trimapGray);
    cv::waitKey();
#endif

    // The protocol we use in the GMM class:
    // 0: sure background
    // 1: possible foreground
    // 2: sure foreground
    // Hence, we are going to invert the trimap and consider the tool scribbles
    // sure background
    trimapGray = (cv::Scalar::all(255) - trimapGray);
    trimapGray /= 255;
    auto initStart = std::chrono::high_resolution_clock::now();
    gmmPtr = new GMM(GMM::Builder().withTrimap(bgra, trimapGray).build());
    auto initFinish = std::chrono::high_resolution_clock::now();
    initElapsed = initFinish - initStart;
    std::cout << "Initialisation time: " << initElapsed.count() << " s\n";
  }

  // Learn GMM
  unsigned int iter = 0;
  auto learnStart = std::chrono::high_resolution_clock::now();
  while (iter < CommandLineReader::getInstance().getNumberOfIterations()) {
    iter++;
    gmmPtr->learnGMMs();
  }
  auto learnFinish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> learnElapsed = learnFinish - learnStart;
  std::cout << "Learning time: " << learnElapsed.count() << " s\n";

  /*
  // Get terminals
  auto termStart = std::chrono::high_resolution_clock::now();
  cv::Mat terminals = gmmPtr->terminals();
  auto termFinish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> termElapsed = termFinish - termStart;
  std::cout << "Terminal computation time: " << termElapsed.count() << " s\n";

  // Binarise terminals
  for (int i = 0; i < terminals.rows; ++i) {
    float *p = terminals.ptr<float>(i);
    for (int j = 0; j < terminals.cols; ++j) {
      if (p[j] > 0.0f)
        p[j] = 1.0f;
      else if (p[j] < 0.0f)
        p[j] = 0.0f;
    }
  }
  cv::Mat binTerminals(terminals.rows, terminals.cols, CV_8UC1);
  terminals.convertTo(binTerminals, CV_8UC1);
  binTerminals *= 255;

  // If we provided a --trimap instead of a --rect, the mask is inverted, so we
  // put it back
  if (!CommandLineReader::getInstance().getTrimapFilePath().empty())
    binTerminals = cv::Scalar::all(255) - binTerminals;

  cv::imshow("Image segmented based on GMM", binTerminals);
  cv::waitKey();
  */

  // Get data term via -logprob bg/fg
  auto termStart = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> logprob = gmmPtr->getLogProb();
  auto termFinish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> termElapsed = termFinish - termStart;
  std::cout << "Terminal computation time: " << termElapsed.count() << " s\n";
  cv::Mat data = logprob[0] - logprob[1];
  for (int i = 0; i < data.rows; ++i) {
    float *p = data.ptr<float>(i);
    for (int j = 0; j < data.cols; ++j) {
      if (p[j] > 0.0f)
        p[j] = 1.0f;
      else if (p[j] < 0.0f)
        p[j] = 0.0f;
    }
  }
  cv::Mat binData(data.rows, data.cols, CV_8UC1);
  data.convertTo(binData, CV_8UC1);
  binData *= 255;

  // If we provided a --trimap instead of a --rect, the mask is inverted, so we
  // put it back
  if (!CommandLineReader::getInstance().getTrimapFilePath().empty())
    binData = cv::Scalar::all(255) - binData;

  cv::imshow("Image segmented via -logprob", binData);
  cv::waitKey();

  // Print total time
  std::chrono::duration<double> elapsed = initElapsed + learnElapsed + termElapsed;
  std::cout << "Total computation time: " << elapsed.count() << " s\n";

  // Release GMM
  delete gmmPtr;

  return EXIT_SUCCESS;
}
