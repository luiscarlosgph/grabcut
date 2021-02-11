/**
 * @brief GMM - Gaussian Mixture Model class that uses the NVIDIA CUDA 7
 *              Tookit GMM as backend.
 *
 * @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   20 August 2019.
 */

#ifndef GMM_H
#define GMM_H

// My includes
#include <grabcut/helper_cuda.h>
#include <grabcut/helper_string.h>
#include <grabcut/util.h>

// Declaration of GMM related functions (implemented in gmm.cu)
cudaError_t GMMAssign(int gmm_N, const float *gmm, int gmm_pitch, const uchar4 *image,
                      int image_pitch, unsigned char *alpha, int alpha_pitch, int width,
                      int height);

cudaError_t GMMInitialize(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch,
                          const uchar4 *image, int image_pitch, unsigned char *alpha,
                          int alpha_pitch, int width, int height);

cudaError_t GMMUpdate(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch, const uchar4 *image,
                      int image_pitch, unsigned char *alpha, int alpha_pitch, int width,
                      int height);

cudaError_t GMMDataTerm(Npp32s *terminals, int terminal_pitch, int gmmN, const float *gmm,
                        int gmm_pitch, const uchar4 *image, int image_pitch,
                        const unsigned char *nmap, int nmapPitch, int width, int height);

cudaError_t GMMBgFgDataTerm(Npp32f *bgTerminals, Npp32f *fgTerminals, int terminal_pitch, int gmmN,
                            const float *gmm, int gmm_pitch, const uchar4 *image, int image_pitch,
                            const unsigned char *nmap, int nmapPitch, int width, int height);

cudaError_t convertTrimapToAlpha(unsigned char *alpha, int width, int height);

cudaError_t convertFourmapToAlpha(unsigned char *alpha, int width, int height,
                                  unsigned char probBg = 64, unsigned char probFg = 128);

/**
 * @class GMM - Gaussian Mixture Model.
 */
class GMM {
 public:
  // Use this class to construct GMM
  class Builder;

  // Constants
  static const int k_chan = 4; // Number of gaussians in each mixture
  static const int k_componentsCount = 2 * k_chan; // Two mixtures: one for the fg, one for the bg

  GMM(const cv::Mat &im, uchar *d_nmap, int nmapPitch, unsigned char *d_alpha,
      int alphaPitch) {
    // Store image and n-map information
    m_d_nmap = d_nmap;
    m_nmapPitch = nmapPitch;
    m_d_alpha = d_alpha;
    m_alphaPitch = alphaPitch;
    m_size.width = im.cols;
    m_size.height = im.rows;

    // Compute blocks and scratch memory needed
    m_blocks = ((m_size.width + 31) / 32) * ((m_size.height + 31) / 32);
    m_scratchGmmSize = (int)(m_blocks * m_gmmPitch * k_componentsCount + m_blocks * k_chan);

    // Allocate resources and upload input image to GPU
    checkCudaErrors(cudaMallocPitch(&m_d_image, &m_imagePitch, m_size.width * sizeof(uchar4), 
                                    m_size.height));
    checkCudaErrors(cudaMemcpy2D(m_d_image, m_imagePitch, im.ptr(),
                                 im.step, m_size.width * sizeof(uchar4),
                                 m_size.height, cudaMemcpyHostToDevice));

#ifndef NDEBUG
    // Download and display the input image for debugging purposes
    cv::Mat output(m_size.height, m_size.width, CV_8UC4);
    checkCudaErrors(cudaMemcpy2D(output.ptr(), output.step, m_d_image, m_imagePitch,
                                 output.cols * sizeof(uchar4), m_size.height, cudaMemcpyDeviceToHost));
    cv::imshow("Debug: input from GPU", output);
    cv::waitKey();
#endif

#ifndef NDEBUG
    // Download and display the nmap for debugging purposes
    cv::Mat tmap(m_size.height, m_size.width, CV_8UC1);
    checkCudaErrors(cudaMemcpy2D(tmap.ptr(), tmap.step, m_d_nmap, m_nmapPitch,
                                 m_size.width * sizeof(uchar), tmap.rows,
                                 cudaMemcpyDeviceToHost));
    cv::imshow("Debug: nmap from GPU", tmap);
    cv::waitKey();
#endif

    // Allocate resoures for GMM properties and scratch memory for intermediate
    // computations
    checkCudaErrors(cudaMalloc(&m_d_scratchMem, m_scratchGmmSize));
    checkCudaErrors(cudaMalloc(&m_d_gmm, m_gmmPitch * k_componentsCount));

    // Allocate doublebuffered alpha: we save the previous one at every
    // learnGMM() step
    checkCudaErrors(cudaMallocPitch(&m_d_prevAlpha, &m_alphaPitch, m_size.width * sizeof(uchar), 
                                    m_size.height));

    // Allocate terminals
    // checkCudaErrors(cudaMallocPitch(&m_d_terminals, &m_pitch,
    //                                m_size.width * sizeof(Npp32f),
    //                                m_size.height));
    checkCudaErrors(cudaMallocPitch(&m_d_bgTerminals, &m_pitch, m_size.width * sizeof(Npp32f), 
                                    m_size.height));
    checkCudaErrors(cudaMallocPitch(&m_d_fgTerminals, &m_pitch, m_size.width * sizeof(Npp32f), 
                                    m_size.height));

#ifndef NDEBUG
    // Download and display alpha for debugging purposes
    cv::Mat alpha(m_size.height, m_size.width, CV_8UC1);
    checkCudaErrors(cudaMemcpy2D(alpha.ptr(), alpha.step, m_d_alpha, m_alphaPitch,
                                 m_size.width * sizeof(uchar), alpha.rows,
                                 cudaMemcpyDeviceToHost));
    cv::imshow("Debug: alpha from GPU", alpha * 255);
    cv::waitKey();
#endif
  }

  /**
   * @brief Destructor: free all the allocated CUDA resources.
   */
  ~GMM() {
    checkCudaErrors(cudaFree(m_d_image));
    checkCudaErrors(cudaFree(m_d_nmap));
    checkCudaErrors(cudaFree(m_d_scratchMem));
    checkCudaErrors(cudaFree(m_d_gmm));
    checkCudaErrors(cudaFree(m_d_alpha));
    checkCudaErrors(cudaFree(m_d_prevAlpha));
    checkCudaErrors(cudaFree(m_d_bgTerminals));
    checkCudaErrors(cudaFree(m_d_fgTerminals));
  }

  /**
   * @brief Initialises two Gaussian mixtures, one for the foreground, one for the background.
   */
  void initGMMs() {
    checkCudaErrors(GMMInitialize(k_componentsCount, m_d_gmm, (float *)m_d_scratchMem,
                                  (int)m_gmmPitch, m_d_image, (int)m_imagePitch, m_d_alpha,
                                  (int)m_alphaPitch, m_size.width, m_size.height));
  }

  /**
   * @bried Modifies alpha putting the number of the component each pixel belongs to.
   */
  void assignGMMs() {
    // Save previous alpha for further comparison (e.g discover if converged)
    checkCudaErrors(cudaMemcpy2D(m_d_prevAlpha, m_alphaPitch, m_d_alpha, m_alphaPitch, 
                                 m_size.width * sizeof(uchar), m_size.height, 
                                 cudaMemcpyDeviceToDevice));
    
    // Given the stored alpha, discover to which component each pixel is linked to
    assignLearning(); 
  }
  
  /**
   * @bried Modifies alpha putting the number of the component each pixel belongs to, but using
   *        the alpha provided as parameter.
   * @param[in] newAlpha Binary mask (0/1) of the size of the input.
   */
  void assignGMMs(const cv::Mat &newAlpha) {
    // Save previous alpha for further comparison (e.g discover if converged)
    checkCudaErrors(cudaMemcpy2D(m_d_prevAlpha, m_alphaPitch, m_d_alpha, m_alphaPitch, 
                                 m_size.width * sizeof(uchar), m_size.height, 
                                 cudaMemcpyDeviceToDevice));

    // Upload newAlpha to GPU for computations but without touching the original one,
    // which is used to keep the original seeds of the user needed by computeDataTerm()
    checkCudaErrors(cudaMemcpy2D(m_d_alpha, m_alphaPitch, newAlpha.ptr(), newAlpha.step, 
                                 m_size.width * sizeof(uchar), m_size.height,
                                 cudaMemcpyHostToDevice));
    
    // Given the new alpha, discover to which component each pixel is linked to
    assignLearning(); 
  }

  /**
   * @brief Perform a learning iteration.
   */
  void learnGMMs() {
    m_iter++;

    // Save previous alpha for further comparison (e.g discover if converged)
    checkCudaErrors(cudaMemcpy2D(m_d_prevAlpha, m_alphaPitch, m_d_alpha, m_alphaPitch,
                                 m_size.width * sizeof(uchar), m_size.height, 
                                 cudaMemcpyDeviceToDevice));

    updateLearning();
  }

  /**
   * @brief Perform a learning iteration but using the new alpha (typically
   * computed by maxflow) as seed.
   */
  void learnGMMs(const cv::Mat &newAlpha) {
    m_iter++;

    // Save previous alpha for further comparison (e.g discover if converged)
    checkCudaErrors(cudaMemcpy2D(m_d_prevAlpha, m_alphaPitch, m_d_alpha, m_alphaPitch,
                                 m_size.width * sizeof(uchar), m_size.height, 
                                 cudaMemcpyDeviceToDevice));

    // Upload newAlpha to GPU for computations but without touching the original one,
    // which is used to keep the original seeds of the user needed by computeDataTerm()
    checkCudaErrors(cudaMemcpy2D(m_d_alpha, m_alphaPitch, newAlpha.ptr(), newAlpha.step, 
                                 m_size.width * sizeof(uchar), m_size.height,
                                 cudaMemcpyHostToDevice));

    updateLearning();
  }

  /**
   * @brief Download current terminals from GPU and get it as a cv::Mat.
   */
  // cv::Mat terminals() {
  //  cv::Mat term(m_size.height, m_size.width, CV_32FC1);
  //  checkCudaErrors(cudaMemcpy2D(term.ptr<float>(), term.step, m_d_terminals,
  //                               (int)m_pitch, m_size.width * sizeof(Npp32f),
  //                               m_size.height, cudaMemcpyDeviceToHost));
  //  return term;
  //}

  /**
   * @brief Get -log probability map for both classes.
   */
  std::vector<cv::Mat> getLogProb() {
    std::vector<cv::Mat> term;
    cv::Mat termBg(m_size.height, m_size.width, CV_32FC1);
    cv::Mat termFg(m_size.height, m_size.width, CV_32FC1);

    // Get log probabilities terminals for fg/bg for all the pixels and
    // download them to CPU memory
    computeDataTerm();

    // Get background terminals
    checkCudaErrors(cudaMemcpy2D(termBg.ptr(), termBg.step, m_d_bgTerminals, (int)m_pitch,
                                 m_size.width * sizeof(float), m_size.height,
                                 cudaMemcpyDeviceToHost));

    // Get foreground terminals
    checkCudaErrors(cudaMemcpy2D(termFg.ptr(), termFg.step, m_d_fgTerminals, (int)m_pitch,
                                 m_size.width * sizeof(float), m_size.height,
                                 cudaMemcpyDeviceToHost));

    term.push_back(termBg);
    term.push_back(termFg);

    return term;
  }

  /**
   * @brief Get a GPU pointer to the segmentation result.
   *
   * @returns a GPU memory pointer.
   */
  unsigned char *getAlphaGPU() { return m_d_alpha; }
  unsigned char *getPrevAlphaGPU() { return m_d_prevAlpha; }

  size_t getAlphaPitch() { return m_alphaPitch; }

  /**
   * @brief Get a GPU pointer to the original image.
   */
  uchar4 *getImageGPU() { return m_d_image; }

  size_t getImagePitch() { return m_imagePitch; }

  Npp8u *getScratchMem() { return m_d_scratchMem; }

 protected:
  // CUDA related attributes
  int m_blocks;

  // GMM variables
  float *m_d_gmm;
  size_t m_gmmPitch = 11 * sizeof(float);
  int m_iter = 0;

  // Data input: image and nmap
  uchar4 *m_d_image;
  size_t m_imagePitch;
  NppiSize m_size;
  unsigned char *m_d_nmap;
  size_t m_nmapPitch;

  // Intermediate computation storage
  unsigned char *m_d_alpha;
  unsigned char *m_d_prevAlpha;
  size_t m_alphaPitch;
  Npp8u *m_d_scratchMem;
  int m_scratchGmmSize;

  // Terminal variables
  // Npp32s *m_d_terminals;
  size_t m_pitch;
  Npp32f *m_d_bgTerminals;
  Npp32f *m_d_fgTerminals;

  // Protected methods
  void assignLearning() {
    checkCudaErrors(GMMAssign(k_componentsCount, m_d_gmm, (int)m_gmmPitch, m_d_image, 
                              (int)m_imagePitch, m_d_alpha, (int)m_alphaPitch, m_size.width, 
                              m_size.height));
  }

  void updateLearning() {
    checkCudaErrors(GMMUpdate(k_componentsCount, m_d_gmm, (float *)m_d_scratchMem, 
                              (int)m_gmmPitch, m_d_image, (int)m_imagePitch, m_d_alpha, 
                              (int)m_alphaPitch, m_size.width, m_size.height));
  }

  void computeDataTerm() {
    checkCudaErrors(GMMBgFgDataTerm(
        m_d_bgTerminals, m_d_fgTerminals, (int)m_pitch, k_componentsCount, m_d_gmm, (int)m_gmmPitch,
        m_d_image, (int)m_imagePitch, m_d_nmap, (int)m_nmapPitch, m_size.width, m_size.height));
  }
};

class GMM::Builder {
 public:
  Builder(int gpuId = -1) {
    // If a GPU was not specified, the fastest is picked
    if (gpuId < 0)
      gpuId = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(gpuId));
  }

  Builder &withRect(const cv::Mat &im, const cv::Rect &cvrect) {
    m_im = im;

    // Convert rect into trimap using GPU
    NppiRect rect;
    rect.x = cvrect.x;
    rect.y = cvrect.y;
    rect.width = cvrect.width;
    rect.height = cvrect.height;
    checkCudaErrors(cudaMallocPitch(&m_d_nmap, &m_nmapPitch, im.cols * sizeof(uchar), im.rows));
    checkCudaErrors(TrimapFromRect(m_d_nmap, (int)m_nmapPitch, rect, im.cols, im.rows));

    // Copy user made nmap to working alpha, the user provided nmap will
    // remain untouched
    checkCudaErrors(cudaMallocPitch(&m_d_alpha, &m_alphaPitch, im.cols * sizeof(uchar), im.rows));
    checkCudaErrors(cudaMemcpy2D(m_d_alpha, m_alphaPitch, m_d_nmap, m_nmapPitch, m_nmapPitch,
                                 im.rows, cudaMemcpyDeviceToDevice));

    // Compute the divisor to convert the nmap to alpha, if the image has
    // sure background divisor will be 128, otherwise 255, then we correct
    // alpha dividing by the previously computed divisor (128 or 255).
    checkCudaErrors(convertTrimapToAlpha(m_d_alpha, im.cols, im.rows));

    return *this;
  }

  Builder &withTrimap(const cv::Mat &im, const cv::Mat &trimap) {
    // CV_Assert(trimap.depth() == CV_8UC1);

    // Keep the image to pass it to GMM
    m_im = im;

    // Allocate and upload trimap to GPU
    checkCudaErrors(
        cudaMallocPitch(&m_d_nmap, &m_nmapPitch, im.cols * sizeof(unsigned char), im.rows));
    checkCudaErrors(cudaMemcpy2D(m_d_nmap, m_nmapPitch, trimap.ptr<unsigned char>(), trimap.step,
                                 im.cols, im.rows, cudaMemcpyHostToDevice));

    // Copy user made nmap to working alpha, the user provided nmap will
    // remain untouched
    checkCudaErrors(cudaMallocPitch(&m_d_alpha, &m_alphaPitch, im.cols, im.rows));
    checkCudaErrors(cudaMemcpy2D(m_d_alpha, m_alphaPitch, m_d_nmap, m_nmapPitch, m_nmapPitch,
                                 im.rows, cudaMemcpyDeviceToDevice));

    // Compute the divisor to convert the nmap to alpha, if the image has
    // sure background divisor will be 128, otherwise 255, then we correct
    // alpha dividing by the previously computed divisor (128 or 255).
    checkCudaErrors(convertTrimapToAlpha(m_d_alpha, m_alphaPitch, im.rows));

    return *this;
  }

  Builder &withFourmap(const cv::Mat &im, const cv::Mat &fourmap) {
    // CV_Assert(fourmap.depth() == CV_8UC1);

    // Keep the image to pass it to GMM
    m_im = im;

    // Allocate and upload fourmap to GPU
    checkCudaErrors(cudaMallocPitch(&m_d_nmap, &m_nmapPitch, im.cols * sizeof(uchar), im.rows));
    checkCudaErrors(cudaMemcpy2D(m_d_nmap, m_nmapPitch, fourmap.ptr<uchar>(), fourmap.step,
                                 im.cols * sizeof(uchar), im.rows, cudaMemcpyHostToDevice));

    // Copy user made nmap to working alpha, the user provided nmap will
    // remain untouched
    checkCudaErrors(cudaMallocPitch(&m_d_alpha, &m_alphaPitch, im.cols * sizeof(uchar), im.rows));
    checkCudaErrors(cudaMemcpy2D(m_d_alpha, m_alphaPitch, m_d_nmap, m_nmapPitch, 
                    im.cols * sizeof(uchar), im.rows, cudaMemcpyDeviceToDevice));
    
    // (0, 64) are converted to 0 and (128, 255) to 1
    checkCudaErrors(convertFourmapToAlpha(m_d_alpha, m_alphaPitch, im.rows));

    return *this;
  }

  GMM build() { 
    return GMM(m_im, m_d_nmap, (int)m_nmapPitch, m_d_alpha, (int)m_alphaPitch); 
  }

 protected:
  cv::Mat m_im;
  uchar *m_d_nmap;
  size_t m_nmapPitch;
  uchar *m_d_alpha;
  size_t m_alphaPitch;
};

#endif
