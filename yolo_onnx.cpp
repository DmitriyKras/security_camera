#include "header.h"


cv::Rect Bbox::toDraw(int width, int height) const {
    int new_x = (x - w / 2) * width;
    int new_y = (y - h / 2) * height;
    int new_w = w * width;
    int new_h = h * height;
    return cv::Rect(new_x, new_y, new_w, new_h);
}


ONNXYOLO::ONNXYOLO(std::string path, cv::Size input_shape, float conf, float iou, int n_classes) :
input_shape(input_shape), conf(conf), iou(iou), n_classes(n_classes) {
    spdlog::info("Setting up model...");
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);  // init enviroment
    sessionOptions = Ort::SessionOptions();  // set session options
    session = Ort::Session(env, path.c_str(), sessionOptions);  // set session
    input_names.reserve(1);  // reserve memory for input names and output names
    output_names.reserve(1);
    spdlog::info("Model prepared. Perform checking...");
    // Input and output nodes checking
    const size_t num_input_nodes = session.GetInputCount();
    if (num_input_nodes != 1)
        throw std::runtime_error(fmt::format("Number of input nodes must be 1, but got {}", num_input_nodes));
    const size_t num_output_nodes = session.GetOutputCount();
    if (num_output_nodes != 1)
        throw std::runtime_error(fmt::format("Number of output nodes must be 1, but got {}", num_output_nodes));
    // Get input name
    auto input_name  = session.GetInputNameAllocated(0, allocator);  // get input name
    input_names.push_back(input_name.get());  // save input name
    input_names_allocated.push_back(std::move(input_name));
    std::cout << fmt::format("Name of model input is {}\n", input_names[0]);
    // Data type checking
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();  // input tensor info
    auto type = tensor_info.GetElementType();  // get data type
    std::cout << fmt::format("Type of model input is {}\n", static_cast<int> (type));
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Data type of model must be float32");
    // Input shape checking
    input_tensor_shape = tensor_info.GetShape();  // get shape of input tensor
    std::cout << fmt::format("Shape of model input is [{}]\n", fmt::join(input_tensor_shape, ", "));
    if (input_tensor_shape.size() != 4)
        throw std::runtime_error(fmt::format("Input shape of model must be 4-dim, bot got {}-dim", input_tensor_shape.size()));
    if (!(input_tensor_shape[0] == 1 && input_tensor_shape[1] == 3 && 
    input_tensor_shape[2] == input_shape.height && input_tensor_shape[3] == input_shape.width))
        throw std::runtime_error(fmt::format("Provided input shape WH is [{}, {}], but actual input shape is [{}, {}]. Number of batch and channels must be 1 and 3.", 
        input_shape.width, input_shape.height, input_tensor_shape[3], input_tensor_shape[2]));
    
    input_tensor_size = tensor_info.GetElementCount();  // get number of input elements
    std::cout << fmt::format("Number of elements in model input is {}\n", input_tensor_size);
    // Get output name
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    output_names.push_back(output_name.get());  // get output name
    output_names_allocated.push_back(std::move(output_name));
    std::move(output_name);
    std::cout << fmt::format("Name of model output is {}\n", output_names[0]);
    // Output shape checking
    type_info = session.GetOutputTypeInfo(0);
    tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_tensor_shape = tensor_info.GetShape();
    std::cout << fmt::format("Shape of model output is [{}]\n", fmt::join(output_tensor_shape, ", "));

    n_boxes = input_shape.width * input_shape.height / 64 * (1 + 1.f/16 + 1.f/4);  // compute output shape based on input shape
    box_width = 4 + n_classes;  // TODO: add support of multiclass
    if (!(output_tensor_shape[0] == 1 && output_tensor_shape[1] == box_width && output_tensor_shape[2] == n_boxes))
        throw std::runtime_error(fmt::format("Output shape of model must be [1, {}, {}], but got [{}]", box_width, n_boxes, fmt::join(output_tensor_shape, ", ")));
    // Prepare buffers
    frame_ptr = new float[input_tensor_size];  // allocate memory for input
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    spdlog::info("All checks passed.");
}


ONNXYOLO::~ONNXYOLO() {
    delete[] frame_ptr;
}


void ONNXYOLO::__preprocess(cv::Mat &frame) {
    cv::Mat buf;  // create buffer
    cv::resize(frame, buf, input_shape);  // resize input frame
    cv::cvtColor(buf, buf, cv::COLOR_BGR2RGB);  // convert frame from BGR to RGB
    buf.convertTo(buf, CV_32FC3, 1. / 255);  // convert from UINT to Float32 and /255
    // Convert memory layout from HWC to CHW
    std::vector<cv::Mat> chw(3);  // another buffer
    for (int i = 0; i < 3; i++)
    {
        chw[i] = cv::Mat(input_shape, CV_32FC1, frame_ptr + i * input_shape.area());
    }
    cv::split(buf, chw);
    // Now frame_ptr has CHW data layout
}


std::vector<Bbox> ONNXYOLO::predict(cv::Mat &frame) {
    __preprocess(frame);  // perform input frame preprocessing
    // Create input buffer
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, frame_ptr, input_tensor_size, input_tensor_shape.data(), 4);
    // Run inference
    std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    float* output_data = output_tensor[0].GetTensorMutableData<float>();  // get pointer to output data memory
    cv::Mat output0 = cv::Mat(cv::Size(n_boxes, box_width), CV_32F, output_data).t();  // convert data layout from [box_width, n_boxes] to [n_boxes, box_width]
    // Prepare vectors to postprocessing results
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    float *data_ptr = (float *)output0.data; // get pointer to transposed memory
    double confidence = 0;
    cv::Point cl;
    // Filter boxes by confidence score
    for (int i = 0; i < n_boxes; i++) {
        // Find max confidence and class id
        cv::Mat scores(1, n_classes, CV_32FC1, data_ptr + 4);
        cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &cl);
        if (confidence > conf) {
            // Fill vectors
            confidences.push_back((float)confidence);
            cv::Rect box = cv::Rect(data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3]);  // xywh
            boxes.push_back(box);
            classes.push_back(cl.x);
        }
        data_ptr += box_width;  // move pointer to the next box
    }
    std::vector<int> nms_result;  // vector for resulting indexes
    cv::dnn::NMSBoxes(boxes, confidences, conf, iou, nms_result);  // perform nms
    std::vector<Bbox> result;  // prepare vector for resulting boxes
    if (nms_result.size() > 0) {
        for (int i : nms_result) {
            cv::Rect box = boxes[i];  // grab box
            result.push_back(Bbox((float)box.x / input_shape.width,
            (float)box.y / input_shape.height, (float)box.width / input_shape.width, (float)box.height / input_shape.height, confidences[i], classes[i]));
        }
    }
    return result;
}


void ONNXYOLO::draw_bboxes(cv::Mat &frame, std::vector<Bbox> &boxes) const {
    if (boxes.size() == 0)  // return if no boxes
        return;
    int width = frame.cols;  // get shape of input frame
    int height = frame.rows;
    for (int i = 0; i < boxes.size(); i++) {
        Bbox box = boxes[i];  // grab bbox
        cv::Rect box2draw = box.toDraw(width, height);
        cv::rectangle(frame, box2draw, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, fmt::format("Class {}: {:.2f}", box.cl, box.conf), cv::Point(box2draw.x, box2draw.y),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
    }
}
