#include "header.h"

void ORBChecker::__log_warning(std::string &message) const {spdlog::warn("Camera " + camera_name + ": " + message);}

void ORBChecker::__log_info(std::string &message) const {spdlog::info("Camera " + camera_name + ": " + message);}


ORBChecker::~ORBChecker() {
    if (verbose == 2) {
        // destroy corresponding windows
        cv::destroyWindow("Base image");
        cv::destroyWindow("Base keypoints");
        cv::destroyWindow("Frame keypoints");
    }
}


void ORBChecker::__check_init(cv::Mat &image) {
    orb->detectAndCompute(base_img, cv::Mat(), base_keypoints, base_descriptors);  // get kpts and descriptors from base frame
    orb->detectAndCompute(image, cv::Mat(), frame_keypoints, frame_descriptors);  // get kpts and descriptors from current frame
    // assert base and current kpts to have at least single point
    // assertion may be failed if image is black or uniform
    if (base_keypoints.size() == 0 or frame_keypoints.size() == 0) {
        if (verbose == 1) {
            std::string msg = "Bad initialization. Starting again.";
            __log_warning(msg);
        }
        base_imgs.clear();  // clear stack to start initialization again
        return;
    }
    matcher->match(base_descriptors, frame_descriptors, matches, cv::Mat());  // match descriptors from base image and current image
    matching_count = matches.size();  // get number of matched descriptors
    // check if base image corresponds to current image
    if (matching_count < (init_matching_ratio * std::min(base_keypoints.size(), frame_keypoints.size()))) {
        if (verbose == 1) {
            std::string msg = "Bad initialization. Starting again.";
            __log_warning(msg);
        }
        base_imgs.clear();  // clear stack to start initialization again
    }
    else {
        // initialization done
        obscured = false;  // set flags
        initialized = true;
        if (verbose == 1) {
            std::string msg = "Initialization done";
            __log_info(msg);
        }
    }
}


void ORBChecker::init_base_img(cv::Mat &frame) {
    if (counter % init_every == 0) {
        base_imgs.push_back(frame);  // append image to stack
        // log info message if verbose
        if (verbose == 1) {
            std::string msg = fmt::format("Images in initial stack: {} out of {}", base_imgs.size(), init_number);
            __log_info(msg);
        }
        // if stack contains enough images perform averaging
        if (base_imgs.size() == init_number) {
            cv::Mat buf = cv::Mat::zeros(size, CV_32F);  // creat float buffer
            for (size_t i = 0; i < init_number; i++)
                buf += base_imgs[i];  // sum images
            base_img = buf / init_number;  // average
            base_img.convertTo(base_img, CV_8U);  // cast to uint8
            if (verbose == 1) {
                std::string msg = "Base image created";
                __log_info(msg);
            }
            __check_init(frame);
        }
    }
}


void ORBChecker::check(cv::Mat &frame) {
    orb->detectAndCompute(frame, cv::Mat(), frame_keypoints, frame_descriptors);  // get kpts and descriptors from current frame
    if (frame_keypoints.size() > 0) {
        // if image is not blacks
        matcher->match(base_descriptors, frame_descriptors, matches, cv::Mat());  // match descriptors from base image and current image
        num_matches = matches.size();  // get number of matched descriptors
        obscured = num_matches / matching_count < threshold;  // obscured if ratio of matched kpts less then threshold
    }
    else {
        // if image is black
        num_matches = 0;
        obscured = true;
    }
    if (verbose == 1) {
        std::string msg = fmt::format("Number of matches: {} out of {} ({:.2f})%", num_matches, (int)matching_count,
        num_matches / matching_count * 100);
        __log_info(msg);
        msg = obscured ? "Is obscured" : "Good condition";
        __log_info(msg);
    }
}


void ORBChecker::update(cv::Mat &frame) {
    // update base image
    matching_count = update_matching_count_ratio * num_matches + (1 - update_matching_count_ratio) * matching_count;
    base_img = frame * update_base_img_ratio + base_img * (1 - update_base_img_ratio);
    orb->detectAndCompute(base_img, cv::Mat(), base_keypoints, base_descriptors);  // get kpts and descriptors from base frame
    if (verbose == 1) {
        std::string msg = "Base image updated";
        __log_info(msg);
    }
}


void ORBChecker::__draw_output(cv::Mat frame) const {
    cv::Mat base_img_copy = base_img.clone();  // copy for base image imshow
    cv::Mat base_kpts_img = base_img.clone();  // copy for kpts drawing
    // draw obscure flag
    cv::String text = obscured ? "Is obscured" : "Good condition";
    cv::putText(base_img_copy, text, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    // draw kpts on frame and base img
    cv::drawKeypoints(frame, frame_keypoints, frame, cv::Scalar(255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::drawKeypoints(base_kpts_img, base_keypoints, base_kpts_img, cv::Scalar(255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    // show images
    cv::imshow("Frame keypoints", frame);
    cv::imshow("Base keypoints", base_kpts_img);
    cv::imshow("Base image", base_img_copy);
}


bool ORBChecker::step(cv::Mat &frame) {
    cv::Mat buf_frame;  // buffer
    cv::cvtColor(frame, buf_frame, cv::COLOR_BGR2GRAY);  // convert to grayscale
    cv::resize(buf_frame, buf_frame, size);  // resize
    if (not initialized)
        init_base_img(buf_frame);  // init base frame if not initialized
    else {
        if (counter % check_every == 0)
            check(buf_frame);  // check frame on obscurness
        
        if (!obscured && counter % update_every == 0)
            update(buf_frame);  // update base frame

        if (verbose == 2)
            __draw_output(buf_frame.clone());  // draw output if verbose
    }
    counter ++;
    return obscured;
}
