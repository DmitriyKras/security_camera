class SecurityCamera
{
private:
    YAML::Node config;  // YAML config file
    CustomVideoCapture *cap;  // video reading
    cv::Size size;  // frame shape
    int period = 1, frame_count = 0;  // counters
    std::vector<Bbox> bboxes;  // yolo results
    ORBChecker orb;  // obscureness checker
    VideoRecording recording;  // record video
    ONNXYOLO *yolo;  // yolo model
    bool obscured = false, print_fps, record_video, checker;  // flag for obscureness
    int direction = 0;
    std::chrono::_V2::system_clock::time_point last_logged = std::chrono::system_clock::now();  // timers

    void __draw_info(cv::Mat &);
    void __print_fps();

public:
    SecurityCamera(std::string);
    void watch();
    void release();
};
