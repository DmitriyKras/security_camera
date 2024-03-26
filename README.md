# security_camera
Pet project with YOLOv8 onnxruntime inference

## Dependencies

[spdlog](https://github.com/gabime/spdlog)

[fmt](https://github.com/fmtlib/fmt)

[OpenCV 4.5.3](https://github.com/opencv/opencv/tree/4.5.3)

[OpenCV contrib 4.5.3](https://github.com/opencv/opencv_contrib/tree/4.5.3)

[onnxruntime] (https://github.com/microsoft/onnxruntime)


## Installation (Linux)

Install dependencies with cmake and `sudo make install`. Then clone this repo, configure config.yaml and run:

`cmake .`

`make`

`./watcher`
