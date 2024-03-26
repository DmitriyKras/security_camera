#include "header.h"


SecurityCamera::SecurityCamera(std::string config_file) {
    config = YAML::LoadFile(config_file);  // load config
    size = cv::Size(config["frame_w"].as<int>(), config["frame_h"].as<int>());  // get frame size
    // Get ORBChecker params and init checker
    int check_every = config["ORB"]["check_every"].as<int>();
    int init_every = config["ORB"]["init_every"].as<int>();
    int init_number = config["ORB"]["init_number"].as<int>();
    int update_every = config["ORB"]["update_every"].as<int>();
    int verbose = config["ORB"]["verbose"].as<int>();
    cv::Size orb_size(config["ORB"]["size_w"].as<int>(), config["ORB"]["size_h"].as<int>());
    float init_matching_ratio = config["ORB"]["init_matching_ratio"].as<float>();
    float update_base_img_ratio = config["ORB"]["update_base_img_ratio"].as<float>();
    float update_matching_count_ratio = config["ORB"]["update_matching_count_ratio"].as<float>();
    float threshold = config["ORB"]["threshold"].as<float>();
    if (config["ORBChecker"].as<bool>()) {
        orb = ORBChecker("0", threshold, orb_size, verbose, init_number, init_every,
        check_every, update_every, init_matching_ratio, update_matching_count_ratio, update_base_img_ratio);
    }
    // VideoRecording
    recording = VideoRecording(config["VIDEO"]["outdir"].as<std::string>(), 60, 600);
    // VideoCapture
    cap = new CustomVideoCapture(config["source"].as<std::string>());
    // YOLO
    float iou = config["YOLO"]["iou"].as<float>();
    float confidence = config["YOLO"]["confidence"].as<float>();
    std::string model_path = config["YOLO"]["model_path"].as<std::string>();
    yolo = new ONNXYOLO(model_path, size, confidence, iou, config["YOLO"]["n_classes"].as<int>());
    // Other params
    print_fps = config["print_fps"].as<bool>();
    record_video = config["record_video"].as<bool>();
    checker = config["ORBChecker"].as<bool>();
}


void SecurityCamera::release() {
    delete yolo;
    cap->release();
    recording.release();
}


void SecurityCamera::__print_fps() {
    if (print_fps) {
        frame_count++;
        auto now = std::chrono::system_clock::now();
        double dif = std::chrono::duration<double, std::milli>(now - last_logged).count() / 1000;
        if (dif > period) {
            float fps = frame_count / dif;
            std::string msg = fmt::format("FPS: {:.2f}", fps);
            spdlog::info(msg);
            last_logged = now;
            frame_count = 0;
        }
    }
}


void SecurityCamera::__draw_info(cv::Mat &frame) {
    if (bboxes.size() > 0)
        yolo->draw_bboxes(frame, bboxes);
    cv::Scalar color(255, 0, 0);
    cv::putText(frame, std::to_string(obscured), cv::Point(10, 210), cv::FONT_HERSHEY_SIMPLEX, 2, color, 2);
}


void SecurityCamera::watch() {
    while (true) {
        __print_fps();
        // Init buffers
        cv::Mat frame;
        cv::Mat rec_frame;
        bool ret = cap->read(frame);  // read frame
        if (!ret)
            continue;
        cv::resize(frame, frame, size);  // resize frame
        frame.copyTo(rec_frame);  // copy frame to record
        // Process yolo
        bboxes = yolo->predict(frame);
        // Process checker
        if (checker)
            obscured = orb.step(frame);
        
        // Process recording
        if (record_video)
            recording.record(rec_frame, bboxes.size() != 0);
        
        // FOR DEBUG //
        
        __draw_info(frame);
        cv::imshow("Camera", frame);
        if ((cv::waitKey(25) & 0xEFFFFF) == 27) {
            cv::destroyWindow("Camera");
            break;
        }
        
    }
}
