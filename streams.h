class CustomVideoCapture
{
private:
    cv::VideoCapture cap;  // default videocapture object
    std::string name;  // name of source
    cv::Mat frame;  // placeholder for frame
    bool ret = false;  // if frame was read
    void __reader();  // threading function
    std::thread th;

public:
    CustomVideoCapture(std::string);
    CustomVideoCapture() {};
    bool read(cv:: Mat &) const;
    void release();
};


class CustomVideoWriter
{
private:
    cv::VideoWriter process;  // default video writer
    bool is_opened = false;  // status
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');  // output file codec
    std::string filename;

public:
    CustomVideoWriter(std::string filename) : filename(filename) {};
    CustomVideoWriter() {};
    void release();
    void write(cv::Mat &);
};


class VideoRecording
{
private:
    std::string outdir;  // out folder to write in
    int timeout, max_file_length;  // params
    std::chrono::_V2::system_clock::time_point timer, stream_time_created;
    bool is_opened = false;  // status
    CustomVideoWriter out_stream;  // writer

    CustomVideoWriter __create_writer();

public:
    VideoRecording(std::string, int, int);
    VideoRecording() {};
    void record(cv::Mat &, bool);
    void release();
};
