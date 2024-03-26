#ifndef HEADER_H
#define HEADER_H

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <limits.h>
#include <unistd.h>
#include <ctime>
#include <thread>
#include <vector>
#include <deque>
#include <map>
#include <tuple>
#include <string>
#include <onnxruntime/include/onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "checkers.h"
#include "yolo_onnx.h"
#include "streams.h"
#include "security_camera.h"

#endif
