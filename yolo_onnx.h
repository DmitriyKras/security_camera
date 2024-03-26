class Bbox
{
public:
    float x, y, w, h, conf;
    int cl;

    Bbox(float x, float y, float w, float h, float conf, int cl) :
    x(x), y(y), w(w), h(h), conf(conf), cl(cl) {};

    cv::Rect toDraw(int, int) const;
};


class ONNXYOLO
{
private:
    int n_classes;
    cv::Size input_shape;  // input shape of model
    Ort::Env env {nullptr};  // enviroment for computing
    Ort::SessionOptions sessionOptions;  // session options
    Ort::Session session {nullptr};  // session for running inference
    std::vector<const char*> input_names;  // vector for input names
    std::vector<int64_t> input_tensor_shape;  // inpur tensor shape
    size_t input_tensor_size;  // number of elements in input tensor
    std::vector<const char*> output_names;  // vector for output names
    float *frame_ptr;  // pointer to frame data
    Ort::MemoryInfo memory_info {nullptr};  // input tensor memory info
    int box_width, n_boxes;  // output data layout params
    float iou, conf;  // iou and confidence threshold for nms
    Ort::AllocatorWithDefaultOptions allocator;  // allocator for getting params
    std::vector<Ort::AllocatedStringPtr> input_names_allocated;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated;

    void __preprocess(cv::Mat &);

public:
    ONNXYOLO(std::string, cv::Size, float, float, int);
    ~ONNXYOLO();
    ONNXYOLO();
    std::vector<Bbox> predict(cv::Mat &);
    void draw_bboxes(cv::Mat &, std::vector<Bbox> &) const;
};
