/**
 * @brief Abstract Graph class oriented to be used for graph-cut like algorithms.
 *
 * @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   10 Sep 2019.
 */
#ifndef GRAPH_H
#define GRAPH_H

// OpenCV
#include <opencv2/core/core.hpp>

class AbstractGraph {
 public:
  /**
   * @brief Basically does nothing because we want to allow for a single graph object to be able to
   *        construct multiple graphs.
   *
   * @param[in] width  Width of the image that we want to convert into a graph.
   * @param[in] height Height of the image.
   */
  AbstractGraph() {}
  virtual ~AbstractGraph(){};

  /**
   * @brief Method to build the graph based on pixel-wise unary and pairwise terms.
   */
  virtual void construct(const cv::Mat &im, const cv::Mat &src, const cv::Mat &sink,
                         const cv::Mat &left, const cv::Mat &topLeft, const cv::Mat &top,
                         const cv::Mat &topRight) = 0;

  /**
   * @brief Solves max-flow problem.
   */
  virtual long maxFlow() = 0;

  /**
   * @returns true if the pixel is in the source/sink partition after max-flow has been solved.
   */
  virtual bool inSourceSegment(const int row, const int col) const = 0;
  virtual bool inSinkSegment(const int row, const int col) const = 0;

 protected:
  /**
   * @brief Creates a new graph node and returns its index.
   */
  // virtual int addVtx() = 0;

  /**
   * @brief Adds two directed edges, vtxIdx -> otherVtxIdx, and otherVtxIdx -> vtxIdx. Both of them
   *        with the same capacity. Typically used to add edges to neighbouring pixels.
   */
  // virtual void addFwdRevEdges(const int vtxIdx, const int otherVtxIdx, const int capacity) = 0;

  /**
   * @brief Adds four edges, from vtxIdx to source and sink, and from source and sink to vtxIdx.
   */
  // virtual void addSrcSinkEdges(const int vtxIdx, const int srcCapacity, const int sinkCapacity) =
  // 0;

  // Class attributes: rows and columns of the image
  int m_rows;
  int m_cols;
};
#endif
