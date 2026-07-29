#ifndef CAFFE_FFI_BACKTRACE_HPP_
#define CAFFE_FFI_BACKTRACE_HPP_

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32) && defined(CAFFE_FFI_ENABLE_BACKTRACE)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <dbghelp.h>
#endif

#if defined(__linux__) && defined(CAFFE_FFI_ENABLE_BACKTRACE)
  #include <execinfo.h>
#endif

namespace caffe_ffi {
namespace backtrace {

static constexpr int kMaxFrames = 32;
static constexpr int kSkipFrames = 2;

inline std::string GetBacktrace(int skip_frames = kSkipFrames, int max_frames = kMaxFrames) {
  if (max_frames <= 0) max_frames = kMaxFrames;
  if (skip_frames < 0) skip_frames = 0;

  std::ostringstream oss;

#if defined(_WIN32) && defined(CAFFE_FFI_ENABLE_BACKTRACE)
  void* frames[kMaxFrames];
  USHORT captured = CaptureStackBackTrace(static_cast<ULONG>(skip_frames),
                                          static_cast<ULONG>(max_frames), frames, nullptr);

  HANDLE process = GetCurrentProcess();
  static bool sym_initialized = false;
  if (!sym_initialized) {
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(process, nullptr, TRUE);
    sym_initialized = true;
  }

  char symbol_buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
  SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_buffer);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen = MAX_SYM_NAME;

  IMAGEHLP_LINE64 line_info;
  line_info.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
  DWORD displacement = 0;

  for (USHORT i = 0; i < captured; ++i) {
    DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
    oss << "  #" << i << " " << frames[i];

    if (SymFromAddr(process, addr, nullptr, symbol)) {
      oss << " in " << symbol->Name;
    }
    if (SymGetLineFromAddr64(process, addr, &displacement, &line_info)) {
      const char* basename = std::strrchr(line_info.FileName, '\\');
      if (!basename) basename = std::strrchr(line_info.FileName, '/');
      basename = basename ? basename + 1 : line_info.FileName;
      oss << " at " << basename << ":" << line_info.LineNumber;
    }
    oss << "\n";
  }
#elif defined(__linux__) && defined(CAFFE_FFI_ENABLE_BACKTRACE)
  void* frames[kMaxFrames];
  int n = ::backtrace(frames, max_frames + skip_frames);
  char** symbols = ::backtrace_symbols(frames, n);
  if (symbols) {
    int idx = 0;
    for (int i = skip_frames; i < n; ++i) {
      oss << "  #" << idx++ << " " << symbols[i] << "\n";
    }
    ::free(symbols);
  }
#else
  oss << "  (backtrace not available: rebuild with CAFFE_FFI_ENABLE_BACKTRACE=ON)\n";
  (void)skip_frames;
#endif

  return oss.str();
}

inline void PrintBacktrace(int skip_frames = kSkipFrames, int max_frames = kMaxFrames) {
  std::string bt = GetBacktrace(skip_frames + 1, max_frames);
  std::cerr << "Backtrace:\n" << bt;
  std::cerr.flush();
}

}  // namespace backtrace
}  // namespace caffe_ffi

#define CAFFE_FFI_BACKTRACE_STR() ::caffe_ffi::backtrace::GetBacktrace(2)
#define CAFFE_FFI_BACKTRACE_STR_SKIP(skip) ::caffe_ffi::backtrace::GetBacktrace(2 + (skip))

#endif  // CAFFE_FFI_BACKTRACE_HPP_
