#ifndef CAFFE_FFI_LOG_HPP_
#define CAFFE_FFI_LOG_HPP_

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace caffe_ffi {
namespace log {

enum class Level {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
};

inline Level& CurrentLevel() {
  static Level level = Level::WARN;
  return level;
}

inline void SetLevel(Level level) {
  CurrentLevel() = level;
}

inline Level GetLevel() {
  return CurrentLevel();
}

inline const char* LevelName(Level level) {
  switch (level) {
    case Level::TRACE: return "TRACE";
    case Level::DEBUG: return "DEBUG";
    case Level::INFO:  return "INFO";
    case Level::WARN:  return "WARN";
    case Level::ERROR: return "ERROR";
    default:           return "UNKNOWN";
  }
}

inline bool ShouldLog(Level level) {
#ifdef CAFFE_FFI_ENABLE_DEBUG_LOG
  return static_cast<int>(level) >= static_cast<int>(CurrentLevel());
#else
  return static_cast<int>(level) >= static_cast<int>(Level::WARN);
#endif
}

class Logger {
 public:
  Logger(Level level, const char* file, int line, const char* func)
      : enabled_(ShouldLog(level)), level_(level) {
    if (enabled_) {
      const char* basename = std::strrchr(file, '\\');
      if (!basename) basename = std::strrchr(file, '/');
      basename = basename ? basename + 1 : file;
      buf_ << "[" << LevelName(level) << "] "
           << basename << ":" << line << " (" << func << ") ";
    }
  }

  ~Logger() {
    if (enabled_) {
      buf_ << "\n";
      if (level_ >= Level::ERROR) {
        std::cerr << buf_.str();
        std::cerr.flush();
      } else {
        std::cout << buf_.str();
        std::cout.flush();
      }
    }
  }

  template <typename T>
  Logger& operator<<(const T& value) {
    if (enabled_) {
      buf_ << value;
    }
    return *this;
  }

 private:
  bool enabled_;
  Level level_;
  std::ostringstream buf_;
};

}  // namespace log
}  // namespace caffe_ffi

#define CAFFE_FFI_LOG(level) \
  ::caffe_ffi::log::Logger(level, __FILE__, __LINE__, __func__)

#define CAFFE_FFI_LOG_TRACE() CAFFE_FFI_LOG(::caffe_ffi::log::Level::TRACE)
#define CAFFE_FFI_LOG_DEBUG() CAFFE_FFI_LOG(::caffe_ffi::log::Level::DEBUG)
#define CAFFE_FFI_LOG_INFO()  CAFFE_FFI_LOG(::caffe_ffi::log::Level::INFO)
#define CAFFE_FFI_LOG_WARN()  CAFFE_FFI_LOG(::caffe_ffi::log::Level::WARN)
#define CAFFE_FFI_LOG_ERROR() CAFFE_FFI_LOG(::caffe_ffi::log::Level::ERROR)

#define CAFFE_FFI_MEM_LOG       CAFFE_FFI_LOG_DEBUG() << "[MEM] "
#define CAFFE_FFI_TENSOR_LOG    CAFFE_FFI_LOG_DEBUG() << "[TENSOR] "
#define CAFFE_FFI_CONTAINER_LOG CAFFE_FFI_LOG_DEBUG() << "[CONTAINER] "
#define CAFFE_FFI_NET_LOG       CAFFE_FFI_LOG_DEBUG() << "[NET] "
#define CAFFE_FFI_LAYER_LOG     CAFFE_FFI_LOG_DEBUG() << "[LAYER] "
#define CAFFE_FFI_BLOB_LOG      CAFFE_FFI_LOG_DEBUG() << "[BLOB] "

#endif  // CAFFE_FFI_LOG_HPP_
