#include "header.h"


int main() {
    std::string config_file = "config.yaml";
    SecurityCamera watcher(config_file);
    watcher.watch();
    watcher.release();
    return 0;
}
