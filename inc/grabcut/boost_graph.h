/**
 * @brief Header with helper classes and typedefs to use the Boost Graph Library.
 *
 * @author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   9 Sep 2019.
 */
#ifndef BOOST_GRAPH_H
#define BOOST_GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/graph_traits.hpp>

// My imports
#include <grabcut/graph.h>

// Boost typedefs
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits_t;
typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS,
    boost::property<
        boost::vertex_name_t, std::string,
        boost::property<
            boost::vertex_index_t, long,
            boost::property<boost::vertex_color_t, boost::default_color_type,
                            boost::property<boost::vertex_distance_t, long,
                                            boost::property<boost::vertex_predecessor_t,
                                                            traits_t::edge_descriptor> > > > >,
    boost::property<
        boost::edge_capacity_t, long,
        boost::property<boost::edge_residual_capacity_t, long,
                        boost::property<boost::edge_reverse_t, traits_t::edge_descriptor> > > >
    graph_t;

/**
 * @class BoostGraph Wrapper of a boost graph for GraphCut-like problems. The aim is to hide all
 *        the boost template complexity and deliver an easy-to-use class.
 */
class BoostGraph : public AbstractGraph {
 public:
  /**
   * @brief Adds two nodes, source (0) and sink (1) to the graph.
   */
  BoostGraph() {
    // Add source and sink nodes
    m_s = boost::add_vertex(m_graph);
    m_t = boost::add_vertex(m_graph);
  }
  ~BoostGraph() {}

  void construct(const cv::Mat &im, const cv::Mat &src, const cv::Mat &sink, const cv::Mat &left,
                 const cv::Mat &topLeft, const cv::Mat &top, const cv::Mat &topRight) {
    m_rows = im.rows;
    m_cols = im.cols;
    for (int i = 0; i < im.rows; i++) {
      const float *leftPtr = left.ptr<float>(i);
      const float *topLeftPtr = topLeft.ptr<float>(i);
      const float *topPtr = top.ptr<float>(i);
      const float *topRightPtr = topRight.ptr<float>(i);
      const float *logProbBgPtr = src.ptr<float>(i);
      const float *logProbFgPtr = sink.ptr<float>(i);

      for (int j = 0; j < im.cols; j++) {
        int vtxIdx = this->addVtx();

        // Set t-weights
        float fromSource = logProbBgPtr[j];
        float toSink = logProbFgPtr[j];
        this->addSrcSinkEdges(vtxIdx, fromSource, toSink);

        // Set n-weights
        if (j > 0) {
          this->addFwdRevEdges(vtxIdx, vtxIdx - 1, leftPtr[j]);
        }
        if (j > 0 && i > 0) {
          this->addFwdRevEdges(vtxIdx, vtxIdx - im.cols - 1, topLeftPtr[j]);
        }
        if (i > 0) {
          this->addFwdRevEdges(vtxIdx, vtxIdx - im.cols, topPtr[j]);
        }
        if (j < im.cols - 1 && i > 0) {
          this->addFwdRevEdges(vtxIdx, vtxIdx - im.cols + 1, topRightPtr[j]);
        }
      }
    }
  }

  /**
   * @brief Solve max-flow problem.
   */
  long maxFlow() { return boykov_kolmogorov_max_flow(m_graph, m_s, m_t); }

  /**
   * @returns true if the given vertex belongs to the source partition after max-flow is solved.
   */
  bool inSourceSegment(const int row, const int col) const {
    // Why adding two? Because the user provides the pixel index, and source and sink nodes
    // are added to the graph before any of the pixels
    return get(boost::vertex_color, m_graph)[m_s] ==
           get(boost::vertex_color, m_graph)[row * m_cols + col + 2];
  }

  /**
   * @returns true if the given vertex belongs to the sink partition after max-flow is solved.
   */
  bool inSinkSegment(const int row, const int col) const {
    // Why adding two? Because the user provides the pixel index, and source and sink nodes
    // are added to the graph before any of the pixels
    return get(boost::vertex_color, m_graph)[m_t] ==
           get(boost::vertex_color, m_graph)[row * m_cols + col + 2];
  }

 protected:
  /**
   * @brief Adds a new node to the graph and returns the index of the node just created.
   */
  int addVtx() { return boost::add_vertex(m_graph); }

  /**
   * @brief Adds to edges with the same capacity: vtxIdx -> otherVtxIdx, otherVtxIdx -> vtxIdx.
   */
  void addFwdRevEdges(const int vtxIdx, const int otherVtxIdx, const int capacity) {
    auto fwd = boost::add_edge(vtxIdx, otherVtxIdx, m_graph).first;
    auto bwd = boost::add_edge(otherVtxIdx, vtxIdx, m_graph).first;

    get(boost::edge_capacity, m_graph)[fwd] = capacity;
    get(boost::edge_capacity, m_graph)[bwd] = capacity;

    get(boost::edge_reverse, m_graph)[fwd] = bwd;
    get(boost::edge_reverse, m_graph)[bwd] = fwd;
  }

  /**
   * @brief Adds to the graph the four edges corresponding to the unary weights.
   */
  void addSrcSinkEdges(const int vtxIdx, const int fromSource, const int toSink) {
    this->addFwdRevEdges(vtxIdx, m_s, fromSource);
    this->addFwdRevEdges(vtxIdx, m_t, toSink);
  }

  // Class attributes
  graph_t m_graph;
  traits_t::vertex_descriptor m_s;
  traits_t::vertex_descriptor m_t;
};

#endif
