/**
 * @brief  Minimal GrabCut segmentation code snippet.
 *
 * @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   11 Feb 2021.
 */

#include <opencv2/core/core.hpp>
#include <grabcut/grabcut.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
  const std::string imagePath = "data/tool_512x409.png"; 
  const std::string trimapPath = "data/trimap_512x409.png";
  const std::string outputPath = "data/output_512x409_trimap_iter_5_gamma_10.png";
  int maxIter = 5;
  float gamma = 10.;

  // Read image and trimap
  cv::Mat im = cv::imread(imagePath);
  cv::Mat imBgra;
  cv::cvtColor(im, imBgra, cv::COLOR_BGR2BGRA);
  cv::Mat trimap = cv::imread(trimapPath, cv::IMREAD_GRAYSCALE);

  // Perform segmentation
  GrabCut gc(maxIter);
  cv::Mat segmentation = gc.estimateSegmentationFromTrimap(imBgra, trimap, gamma);

  // Save segmentation
  cv::imwrite(outputPath, segmentation);

  return 0;
}
