/**
 * @class CommandLineReader reads the command line string provided by the user,
 * parses it and
 *        extracts the relevant parameters.
 *
 * @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
 * @date   8 Apr 2015.
 */

#ifndef COMMAND_LINE_READER_H
#define COMMAND_LINE_READER_H

#include <exception>
#include <iostream>
#include <string>

// My includes
#include <cxxopts.h>

class CommandLineReader {
 public:
  // Program name
  const std::string programName = "GMM and GrabCut";

  const std::string kUsageMsg =
      "\nBinary Gaussian Mixture Model segmentation.\n"
      "\nExamples of valid execution:\n"
      "   $ ./main --help\n"
      "   $ ./main --input tool.png --output output.png --niter 5 --rect "
      "tool_rect.png\n"
      "   $ ./main --input tool.png --output output.png --niter 5 --fourmap "
      "tool_fourmap.png\n"
      "   $ ./main --input tool.png --output output.png --niter 5 --trimap "
      "tool_trimap.png --probaout probaout.png\n"
      "   $ ./main --input tool.png --output output.png --niter 5 --trimap "
      "tool_trimap.png\n";

  // Singleton: only one command line
  static CommandLineReader &getInstance() {
    static CommandLineReader instance;
    return instance;
  }

  // Parameter getters
  std::string getInputFilePath() const { return m_inputPath; }
  std::string getOutputFilePath() const { return m_outputPath; }
  std::string getRectFilePath() const { return m_rectPath; }
  std::string getTrimapFilePath() const { return m_trimapPath; }
  std::string getFourmapFilePath() const { return m_fourmapPath; }
  std::string getFgMapFilePath() const { return m_fgMapPath; }
  std::string getBgMapFilePath() const { return m_bgMapPath; }
  uint32_t getNumberOfIterations() const { return m_iter; }
  double getGamma() const { return m_gamma; }

  bool processCmdLineOptions(int argc, char **argv) {
    cxxopts::Options options(programName, kUsageMsg);

    options.add_options()("h,help", "Prints this help message.")(
        "i,input", "Path to the input file.", cxxopts::value<std::string>())(
        "o,output", "Path to the output file.", cxxopts::value<std::string>())(
        "r,rect", "Black image with only one white rectangle in the area of interest.",
        cxxopts::value<std::string>())(
        "t,trimap", "Trimap where background 0, unknown 128, and foreground is 255.",
        cxxopts::value<std::string>())(
        "f,fourmap",
        "Fourmap where background 0, probably background 64, probably foreground "
        "128, and foreground is 255.",
        cxxopts::value<std::string>())(
        "x,fgmap", "Background probability map, the range [0, 255] will be mapped to [0, 1].",
        cxxopts::value<std::string>())(
        "y,bgmap", "Foreground probability map, the range [0, 255] will be mapped to [0, 1].",
        cxxopts::value<std::string>())("g,gamma", "GrabCut gamma. A typical value is 50.0.",
                                       cxxopts::value<double>())(
        "n,niter", "Maximum number of iterations to execute learning.", cxxopts::value<int>());

    // Parse command line options
    auto result = options.parse(argc, argv);

    // Print help if the user asks for it
    if (result.count("help")) {
      printUsage(std::cout);
      return false;
    }

    // Mandatory parameters
    if (result.count("input")) {
      m_inputPath = result["input"].as<std::string>();
    } else {
      std::cerr << "An --input image file is needed." << std::endl;
      return false;
    }

    if (result.count("output")) {
      m_outputPath = result["output"].as<std::string>();
    } else {
      std::cerr << "An --output image file is needed." << std::endl;
      return false;
    }

    // Optional parameters
    if (result.count("niter")) m_iter = result["niter"].as<int>();
    if (result.count("gamma")) m_gamma = result["gamma"].as<double>();
    if (result.count("rect")) m_rectPath = result["rect"].as<std::string>();
    if (result.count("trimap")) m_trimapPath = result["trimap"].as<std::string>();
    if (result.count("fourmap")) m_fourmapPath = result["fourmap"].as<std::string>();
    if (result.count("fgmap")) m_fgMapPath = result["fgmap"].as<std::string>();
    if (result.count("bgmap")) m_bgMapPath = result["bgmap"].as<std::string>();

    return true;
  }

  void printUsage(std::ostream &stream) const { stream << kUsageMsg << std::endl; }

 protected:
  CommandLineReader() {}
  CommandLineReader(CommandLineReader const &) = delete;
  void operator=(CommandLineReader const &) = delete;

  std::string m_inputPath = "";
  std::string m_outputPath = "";
  std::string m_rectPath = "";
  std::string m_trimapPath = "";
  std::string m_fourmapPath = "";
  std::string m_fgMapPath = "";
  std::string m_bgMapPath = "";
  unsigned int m_iter = 0;
  double m_gamma = -1;
};

#endif  // COMMAND_LINE_READER_H
