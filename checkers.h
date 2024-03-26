class ORBChecker
{
private:
    std::string camera_name;  // name of camera stream
    float threshold, init_matching_ratio, update_base_img_ratio, update_matching_count_ratio;  // hyperparameters
    int verbose, init_number, init_every, check_every, update_every;
    cv::Size size;  // frame size

    bool initialized = false, obscured = true;  // states
    int counter = 0, num_matches = 0;  // counter, ideal number of matched points, last matched number of points
    float matching_count = -1;

    std::vector <cv::Mat> base_imgs;  // vector for base images
    cv::Mat base_img;  // resulting base image
    std::vector <cv::KeyPoint> frame_keypoints, base_keypoints;  // vector to store keypoints from base image and current image
    cv::Mat frame_descriptors, base_descriptors;  // descriptors for current image and base image
    cv::Ptr<cv::Feature2D> orb;  // placeholder for ORB detector
    cv::Ptr<cv::BFMatcher> matcher;  // placeholder for BruteForce matcher
    std::vector<cv::DMatch> matches;  // vector to store matches

    void __log_warning(std::string &message) const;
    void __log_info(std::string &message) const;
    void __draw_output(cv::Mat frame) const;
    void __check_init(cv::Mat &image);
    void init_base_img(cv::Mat &frame);
    void check(cv::Mat &frame);
    void update(cv::Mat &frame);

public:
    ORBChecker(std::string camera_name, float threshold = 0.7, cv::Size size = cv::Size(426, 240), int verbose = 1,
    int init_number = 10, int init_every = 3, int check_every = 3, int update_every = 15, 
    float init_matching_ratio = 0.6, float update_matching_count_ratio = 0.4, float update_base_img_ratio = 0.8) :
    camera_name(camera_name),
    threshold(threshold),
    size(size),
    verbose(verbose),
    init_number(init_number),
    init_every(init_every),
    check_every(check_every),
    update_every(update_every),
    init_matching_ratio(init_matching_ratio),
    update_matching_count_ratio(update_matching_count_ratio),
    update_base_img_ratio(update_base_img_ratio),
    orb(cv::ORB::create()),
    matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true)),
    base_img(cv::Mat::zeros(size, CV_8U)) {};

    ~ORBChecker();
    ORBChecker() {};

    bool step(cv::Mat &frame);
    void set(bool status) {obscured = status;}
};
