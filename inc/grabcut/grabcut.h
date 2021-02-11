/**
 * @brief Header only GrabCut implementation based on the NVIDIA GMM and OpenCV
 * maxflow.
 *
 * @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   22 August 2019.
 */

#ifndef GRABCUT_H
#define GRABCUT_H

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

// CUDA
#include <cuda_runtime.h>
#include <npp.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Debug mode
#ifndef NDEBUG
#include <chrono>
#include <iostream>
#endif

// My includes
#include <grabcut/boost_graph.h>
#include <grabcut/gmm.h>
#include <grabcut/opencv_graph.h>

// Declaration of border matting (implemented in util.cu)
cudaError_t applyMatte(int mode, uchar4 *result, int resultPitch, const uchar4 *image,
                       int imagePitch, const unsigned char *matte, int mattePitch, int width,
                       int height);

// Declaration of pairwise weight estimation (implemented in util.cu)
cudaError_t edgeCues(float alpha, const uchar4 *image, int image_pitch, Npp32f *left,
                     Npp32f *topLeft, Npp32f *top, Npp32f *topRight, int pitch, int width,
                     int height, float *scratchMem);

// The function that detects convergence
cudaError_t segmentationConverged(bool &result, int *d_changed, uchar *alphaOld, uchar *alphaNew,
                                  int alphaPitch, int width, int height);

// Function to convert probability maps into negative log cost images
cudaError_t convertToPenalty(cv::Mat &output, const cv::Mat &proba);

/**
 * @class GrabCut implementation.
 */
class GrabCut {
 public:

  /**
   * @param[in] gpuId Index of the GPU to be used. If it is not specified the
   *                  fastest GPU will be chosen (make sure it is available ;).
   */
  GrabCut(int maxIter = 10, int gpuId = -1) { 
    m_maxIter = maxIter; 
    m_gpuId = gpuId;
  }

  /**
   * @brief Calculate beta - parameter of GrabCut algorithm.
   *        beta = 1 / (2 * avg(sqr(||color[i] - color[j]||)))
   */
  static double calcBeta(const cv::Mat &im) {
    double beta = 0;
    for (int y = 0; y < im.rows; y++) {
      for (int x = 0; x < im.cols; x++) {
        cv::Vec3d color = im.at<cv::Vec3b>(y, x);

        // Left
        if (x > 0) {
          cv::Vec3d diff = color - (cv::Vec3d)im.at<cv::Vec3b>(y, x - 1);
          beta += diff.dot(diff);
        }

        // Upleft
        if (y > 0 && x > 0) {
          cv::Vec3d diff = color - (cv::Vec3d)im.at<cv::Vec3b>(y - 1, x - 1);
          beta += diff.dot(diff);
        }

        // Up
        if (y > 0) {
          cv::Vec3d diff = color - (cv::Vec3d)im.at<cv::Vec3b>(y - 1, x);
          beta += diff.dot(diff);
        }

        // Upright
        if (y > 0 && x < im.cols - 1) {
          cv::Vec3d diff = color - (cv::Vec3d)im.at<cv::Vec3b>(y - 1, x + 1);
          beta += diff.dot(diff);
        }
      }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
      beta = 0;
    else
      beta = 1.f / (2 * beta / (4 * im.cols * im.rows - 3 * im.cols - 3 * im.rows + 2));

    return beta;
  }

  /**
   * @brief Computation of the pairwise term using CUDA.
   *
   */
  static void calcNWeightsGPU(const uchar4 *d_image, int imagePitch, int width, int height,
                              float *d_scratchMem, cv::Mat &left, cv::Mat &topLeft, cv::Mat &top,
                              cv::Mat &upright, double gamma) {
    Npp32f *d_left;
    Npp32f *d_topLeft;
    Npp32f *d_top;
    Npp32f *d_topRight;
    size_t pitch;

    // Allocate GPU memory for the neighbour matrices
    checkCudaErrors(cudaMallocPitch(&d_left,    &pitch, width * sizeof(Npp32f), height));
    checkCudaErrors(cudaMallocPitch(&d_topLeft, &pitch, width * sizeof(Npp32f), height));
    checkCudaErrors(cudaMallocPitch(&d_top,     &pitch, width * sizeof(Npp32f), height));
    checkCudaErrors(cudaMallocPitch(&d_topRight,&pitch, width * sizeof(Npp32f), height));
    
    // Compute pairwise weights
    checkCudaErrors(edgeCues(gamma, d_image, imagePitch, d_left, d_topLeft, d_top, d_topRight,
                             (int)pitch, width, height, d_scratchMem));

    // Allocate CPU memory for the neighbour matrices
    left.create(height, width, CV_32FC1);
    topLeft.create(height, width, CV_32FC1);
    top.create(height, width, CV_32FC1);
    upright.create(height, width, CV_32FC1);

    // Download pairwise weights to CPU
    checkCudaErrors(cudaMemcpy2D(left.ptr(), left.step, d_left, (int)pitch,
                                 width * sizeof(float), height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(topLeft.ptr(), topLeft.step, d_topLeft, (int)pitch,
                                 width * sizeof(float), height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(top.ptr(), top.step, d_top, (int)pitch,
                                 width * sizeof(float), height, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(upright.ptr(), upright.step, d_topRight, (int)pitch,
                                 width * sizeof(float), height, cudaMemcpyDeviceToHost));

    // Free GPU space
    checkCudaErrors(cudaFree(d_left));
    checkCudaErrors(cudaFree(d_topLeft));
    checkCudaErrors(cudaFree(d_top));
    checkCudaErrors(cudaFree(d_topRight));
  }

  /**
   * @brief Computes the segmentation from a trimap built based on the given
   * rect. A trimap is a CV_8UC1 image with three
   *        possible values for a pixel: 0 (sure background), 128 (unknown), 255
   * (sure foreground).
   */
  cv::Mat estimateSegmentationFromRect(const cv::Mat &im, const cv::Rect &rect,
                                       const double gamma = 50.0) const {
    CV_Assert(im.channels() == 4);
#ifndef NDEBUG
    auto initStart = std::chrono::high_resolution_clock::now();
#endif

    // Build GMM
    GMM gmm(GMM::Builder(m_gpuId).withRect(im, rect).build());

#ifndef NDEBUG
    // Compute and print elapsed time
    auto initEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> initElapsed = initEnd - initStart;
    std::cout << "initElapsed: " << initElapsed.count() << " s\n";
#endif

    return estimateSegmentationFromGMM(im, gmm, gamma);
  }

  /**
   * @brief Computes the segmentation from a trimap. A trimap is a CV_8UC1 image
   * with three
   *        possible values for a pixel: 0 (sure background), 128 (unknown), 255
   * (sure foreground).
   */
  cv::Mat estimateSegmentationFromTrimap(const cv::Mat &im, const cv::Mat &trimap,
                                         const double gamma = 50.0) const {
    // The input image must be BGRA
    CV_Assert(im.channels() == 4);

#ifndef NDEBUG
    auto initStart = std::chrono::high_resolution_clock::now();
#endif

    // Build GMM
    GMM gmm(GMM::Builder(m_gpuId).withTrimap(im, trimap).build());

#ifndef NDEBUG
    // Compute and print elapsed time
    auto initEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> initElapsed = initEnd - initStart;
    std::cout << "initElapsed: " << initElapsed.count() << " s\n";
#endif

    return estimateSegmentationFromGMM(im, gmm, gamma);
  }

  /**
   * @brief Computes the segmentation from a fourmap. A fourmap is a CV_8UC1 image with four
   *        possible values for a pixel: 0 (sure background), 64 (probably background), 128
   *        (probably foreground), 255 (sure foreground).
   */
  cv::Mat estimateSegmentationFromFourmap(const cv::Mat &im, const cv::Mat &fourmap,
                                          const double gamma = 50.0) const {
    // The input image must be BGRA
    CV_Assert(im.channels() == 4);

#ifndef NDEBUG
    auto initStart = std::chrono::high_resolution_clock::now();
#endif

    // Build GMM
    GMM gmm(GMM::Builder(m_gpuId).withFourmap(im, fourmap).build());

#ifndef NDEBUG
    // Compute and print elapsed time
    auto initEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> initElapsed = initEnd - initStart;
    std::cout << "Time to construct GMM: " << initElapsed.count() << " s\n";
#endif

    return estimateSegmentationFromGMM(im, gmm, gamma);
  }

  /**
   * @brief Given an image and a GMM returns a [0/1] segmentation as a cv::Mat
   *        (CV_8UC1).
   *
   *    @param[in] im     Input BGRA image (must be CV_8UC4).
   *    @param[in] gmm    GMM (constructed but nothing else).
   */
  cv::Mat estimateSegmentationFromGMM(const cv::Mat &im, GMM &gmm,
                                      const double gamma = 50.0) const {
#ifndef NDEBUG
    // Timing variables
    std::chrono::duration<double> pairwiseElapsed;
    std::chrono::duration<double> getLogProbElapsed;
    std::chrono::duration<double> graphConstructElapsed;
    std::chrono::duration<double> maxFlowElapsed;
    std::chrono::duration<double> getResultElapsed;
    std::chrono::duration<double> learnElapsed;
#endif

    // Create the matrix for the final segmentation result
    cv::Mat alpha(im.rows, im.cols, CV_8UC1);

#ifndef NDEBUG
    auto pairwiseStart = std::chrono::high_resolution_clock::now();
#endif
    // Estimate pairwise term weights from the image
    cv::Mat leftW, upleftW, upW, uprightW;
    calcNWeightsGPU(gmm.getImageGPU(), (int)gmm.getImagePitch(), im.cols, im.rows,
                    (float *)gmm.getScratchMem(), leftW, upleftW, upW, uprightW, gamma);
#ifndef NDEBUG
    auto pairwiseEnd = std::chrono::high_resolution_clock::now();
    pairwiseElapsed = pairwiseEnd - pairwiseStart;
#endif

    // Initialise GMM mixtures based on the prior provided
    gmm.initGMMs();
    gmm.learnGMMs();
    
    // GrabCut iterative minimisation
    int iter = 0;
    bool converged = false;
    while (iter < m_maxIter && !converged) {
      iter++;

      // Assign GMM components to pixels and learn GMMs using the current alpha as seed
      //if (iter == 1)
      //  gmm.assignGMMs();
      //else 
      //  gmm.assignGMMs(alpha);

#ifndef NDEBUG
      auto getLogProbStart = std::chrono::high_resolution_clock::now();
#endif
      // Get unary term from GMM
      std::vector<cv::Mat> logProb = gmm.getLogProb();
#ifndef NDEBUG
      auto getLogProbEnd = std::chrono::high_resolution_clock::now();
      getLogProbElapsed = getLogProbEnd - getLogProbStart;
#endif

#ifndef NDEBUG
      auto graphConstructStart = std::chrono::high_resolution_clock::now();
#endif
      // Build graph estimate segmentation using min cut
      OpenCVGraph graph;
      graph.construct(im, logProb[0], logProb[1], leftW, upleftW, upW, uprightW);
#ifndef NDEBUG
      auto graphConstructEnd = std::chrono::high_resolution_clock::now();
      graphConstructElapsed = graphConstructEnd - graphConstructStart;
#endif

#ifndef NDEBUG
      auto maxFlowStart = std::chrono::high_resolution_clock::now();
#endif
      // Solve min-cut max-flow
      graph.maxFlow();
#ifndef NDEBUG
      auto maxFlowEnd = std::chrono::high_resolution_clock::now();
      maxFlowElapsed = maxFlowEnd - maxFlowStart;
#endif

#ifndef NDEBUG
      auto getResultStart = std::chrono::high_resolution_clock::now();
#endif
      // Update alpha
      for (int i = 0; i < alpha.rows; i++) {
        for (int j = 0; j < alpha.cols; j++) {
          int vtxIdx = i * alpha.cols + j;
          alpha.data[vtxIdx] = graph.inSourceSegment(i, j) ? 1 : 0;
        }
      }

      // Update convergence flag
      checkCudaErrors(segmentationConverged(converged, (int *)gmm.getScratchMem(),
                                            gmm.getPrevAlphaGPU(), gmm.getAlphaGPU(),
                                            (int)gmm.getAlphaPitch(), im.cols, im.rows));
#ifndef NDEBUG
      auto getResultEnd = std::chrono::high_resolution_clock::now();
      getResultElapsed = getResultEnd - getResultStart;
#endif
        
#ifndef NDEBUG
      auto learnStart = std::chrono::high_resolution_clock::now();
#endif
      // Learn GMMs from data based on the new alpha
      gmm.learnGMMs(alpha);
#ifndef NDEBUG
      auto learnEnd = std::chrono::high_resolution_clock::now();
      learnElapsed = learnEnd - learnStart;
#endif

    }

#ifndef NDEBUG
    // Compute and print timing results
    // auto totalEnd = std::chrono::high_resolution_clock::now();
    // std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Max. iterations: " << m_maxIter << std::endl;
    std::cout << "Time to estimate pairwise term for the image: " << pairwiseElapsed.count()
              << " s\n";
    std::cout << "Time to estimate the unary term (getLogProb): " << getLogProbElapsed.count()
              << " s\n";
    std::cout << "Time to build the graph: " << graphConstructElapsed.count() << " s\n";
    std::cout << "Time to perform max-flow: " << maxFlowElapsed.count() << " s\n";
    std::cout << "Time to get the result from the graph: " << getResultElapsed.count()
              << " s\n";
    // std::cout << "Total elapsed: " << totalElapsed.count() << " s\n";
#endif
    
    return alpha;
  }

  /** * @brief This method assumes that the unary term is fixed and given and runs just one
   *        maxflow iteration to obtain the segmentation.
   */
  cv::Mat estimateSegmentationFromUnary(const cv::Mat &im, const cv::Mat &unaryBg,
                                        const cv::Mat &unaryFg, const double gamma = 50.0) const {
#ifndef NDEBUG
    // Timing variables
    std::chrono::duration<double> graphConstructElapsed;
    std::chrono::duration<double> maxFlowElapsed;
    std::chrono::duration<double> learnElapsed;
#endif
    
    // Create the matrix for the final segmentation result
    cv::Mat alpha(im.rows, im.cols, CV_8UC1);

    // Upload image to GPU
    uchar4 *d_image;
    size_t imagePitch;
    checkCudaErrors(cudaMallocPitch(&d_image, &imagePitch, im.cols * sizeof(uchar4), im.rows));
    checkCudaErrors(cudaMemcpy2D(d_image, imagePitch, im.ptr(), im.step,
                                 im.cols * sizeof(uchar4), im.rows, cudaMemcpyHostToDevice));
#ifndef NDEBUG
    // Download and display the input image for debugging purposes
    cv::Mat output(im.rows, im.cols, CV_8UC4);
    checkCudaErrors(cudaMemcpy2D(output.ptr(), output.step, d_image, imagePitch,
                                 output.cols * sizeof(uchar4), im.rows, cudaMemcpyDeviceToHost));
    cv::imshow("Debug: input from GPU", output);
    cv::waitKey();
#endif

    // Allocate GPU scratch memory
    int blocks = ((im.cols + 31) / 32) * ((im.rows + 31) / 32);
    int scratchSize = (int)(blocks * 11 * sizeof(float) * 8 + blocks * 4);
    Npp8u *d_scratchMem;
    checkCudaErrors(cudaMalloc(&d_scratchMem, scratchSize));

#ifndef NDEBUG
    auto pairwiseStart = std::chrono::high_resolution_clock::now();
#endif
    // Estimate pairwise term weights from the image
    cv::Mat leftW, upleftW, upW, uprightW;
    calcNWeightsGPU(d_image, (int)imagePitch, im.cols, im.rows, (float *)d_scratchMem, leftW,
                    upleftW, upW, uprightW, gamma);
#ifndef NDEBUG
    auto pairwiseEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pairwiseElapsed = pairwiseEnd - pairwiseStart;
#endif

#ifndef NDEBUG
    auto graphConstructStart = std::chrono::high_resolution_clock::now();
#endif
    // Build graph
    OpenCVGraph graph;
    graph.construct(im, unaryBg, unaryFg, leftW, upleftW, upW, uprightW);
#ifndef NDEBUG
    auto graphConstructEnd = std::chrono::high_resolution_clock::now();
    graphConstructElapsed = graphConstructEnd - graphConstructStart;
#endif

#ifndef NDEBUG
    auto maxFlowStart = std::chrono::high_resolution_clock::now();
#endif
    // Solve min-cut max-flow
    graph.maxFlow();
#ifndef NDEBUG
    auto maxFlowEnd = std::chrono::high_resolution_clock::now();
    maxFlowElapsed = maxFlowEnd - maxFlowStart;
#endif

#ifndef NDEBUG
    auto learnStart = std::chrono::high_resolution_clock::now();
#endif
    // Update alpha
    for (int i = 0; i < alpha.rows; i++) {
      for (int j = 0; j < alpha.cols; j++) {
        int vtxIdx = i * alpha.cols + j;
        alpha.data[vtxIdx] = graph.inSourceSegment(i, j) ? 1 : 0;
      }
    }

#ifndef NDEBUG
    auto learnEnd = std::chrono::high_resolution_clock::now();
    learnElapsed = learnEnd - learnStart;
#endif

#ifndef NDEBUG
    std::cout << "Time to estimate pairwise term for the image: " << pairwiseElapsed.count()
              << " s\n";
    std::cout << "Time to build the graph: " << graphConstructElapsed.count() << " s\n";
    std::cout << "Time to perform max-flow: " << maxFlowElapsed.count() << " s\n";
    std::cout << "Time to get the result from the graph: " << learnElapsed.count() << " s\n";
#endif

    // Free GPU memory allocated for input image
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_scratchMem));

    return alpha;
  }

  /**
   * @brief Segmentation based on a graph cut using two probability maps as inputs. The probability
   *        maps typically come from a softmax output of a CNN.
   */
  cv::Mat estimateSegmentationFromProba(const cv::Mat &im, const cv::Mat &probaBg,
                                        const cv::Mat &probaFg, const double gamma = 50.0) const {
    CV_Assert(im.channels() == 4);
#ifndef NDEBUG
    // Display the probability maps
    cv::imshow("Debug: bg proba map", probaBg);
    cv::waitKey();
    cv::imshow("Debug: fg proba map", probaFg);
    cv::waitKey();
#endif

    // Convert probabilities into penalties (integer capacities for max-flow)
    cv::Mat unaryBg, unaryFg;
    checkCudaErrors(convertToPenalty(unaryBg, probaBg));
    checkCudaErrors(convertToPenalty(unaryFg, probaFg));

    return estimateSegmentationFromUnary(im, unaryBg, unaryFg, gamma);
  }

 protected:
  int m_maxIter;
  int m_gpuId;
};

// Python wrapper of the class GrabCut
#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
static void init_ar() {
#endif
  Py_Initialize();
  import_array();
  //return NUMPY_IMPORT_ARRAY_RETVAL;
  return NULL;
}

BOOST_PYTHON_MODULE(grabcut) {
  // cv::Mat <=> Numpy converter
  init_ar();
  boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
  pbcvt::matFromNDArrayBoostConverter();

  // GrabCut class wrapper
  boost::python::class_<GrabCut>("GrabCut", boost::python::init<int>())
      .def("estimateSegmentationFromTrimap", &GrabCut::estimateSegmentationFromTrimap)
      .def("estimateSegmentationFromFourmap", &GrabCut::estimateSegmentationFromFourmap)
      .def("estimateSegmentationFromProba", &GrabCut::estimateSegmentationFromProba);
}

#endif
