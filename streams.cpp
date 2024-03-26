#include "header.h"


CustomVideoCapture::CustomVideoCapture(std::string name) : 
name(name), cap(cv::VideoCapture(name)) {
    th = std::thread([=] {__reader();});
}


void CustomVideoCapture::__reader() {
    while (true) {
        ret = cap.read(frame);  // try to read frame
        if (!ret) {
            std::this_thread::sleep_for(std::chrono::seconds(5));  // slep for 5 seconds if no frame
            cap.release();
            std::string msg = fmt::format("Attempt to reconnect to camera {}", name);
            spdlog::warn(msg);
            cap = cv::VideoCapture(name);  // reopen stream
        }
    }
}


bool CustomVideoCapture::read(cv::Mat &image) const {
    frame.copyTo(image);
    return ret;
}


void CustomVideoCapture::release() {cap.release();}


void CustomVideoWriter::release() {
    if (is_opened)
        process.release();  // release if exists and change status
        is_opened = false;
}


void CustomVideoWriter::write(cv::Mat &frame) {
    if (!is_opened) {
        cv::Size size = frame.size();  // get frame size
        process = cv::VideoWriter(filename, fourcc, 10, size);  // create writer and change status
        is_opened = true;
    }
    process.write(frame);  // write frame
}


VideoRecording::VideoRecording(std::string outdir, int person_timeout = 60, int max_file_length = 600) : 
outdir(outdir), timeout(person_timeout * 1000), max_file_length(max_file_length * 1000) {
    if (!std::filesystem::exists(outdir))
        std::filesystem::create_directory(outdir);  // create outdir if not exists
}


CustomVideoWriter VideoRecording::__create_writer() {
    // get hostname
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    // get current datetime
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%d-%m-%Y-%H-%M-%S", timeinfo);
    std::string datetime(buffer);
    // get current time stream created
    stream_time_created = std::chrono::system_clock::now();
    // create writer
    std::string path = fmt::format("{}/{}-{}.mov", outdir, hostname, datetime);
    return CustomVideoWriter(path);
}


void VideoRecording::release() {
    if (is_opened) {
        out_stream.release();
        is_opened = false;
    }
}


void VideoRecording::record(cv::Mat &frame, bool is_detected) {
    if (is_detected)
        timer = std::chrono::system_clock::now();

    auto now = std::chrono::system_clock::now();
    if (std::chrono::duration<double, std::milli>(now - timer).count() < timeout) {
        if (!is_opened) {
            out_stream = __create_writer();
            is_opened = true;
        }
        out_stream.write(frame);
        auto now = std::chrono::system_clock::now();
        if (std::chrono::duration<double, std::milli>(now - stream_time_created).count() > max_file_length)
            release();
    }
    else
        release();
}
