#include "include/ung_prof_log.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sys/types.h>
#include <unistd.h>

namespace ANNS
{
namespace
{

constexpr const char *kProfLogDir = "/home/graphdb/Codes/FilterVectorResultsCUDA/prof";

std::mutex g_prof_mtx;

} // namespace

const std::string &prof_log_path()
{
   static std::string path = [] {
      namespace fs = std::filesystem;
      std::error_code ec;
      fs::create_directories(kProfLogDir, ec);

      auto now = std::chrono::system_clock::now();
      std::time_t t = std::chrono::system_clock::to_time_t(now);
      std::tm tm{};
      localtime_r(&t, &tm);

      char ts[32];
      std::snprintf(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d",
                    tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                    tm.tm_hour, tm.tm_min, tm.tm_sec);

      char fname[128];
      std::snprintf(fname, sizeof(fname), "ung_prof_%s_%d.log", ts, static_cast<int>(getpid()));
      return (fs::path(kProfLogDir) / fname).string();
   }();
   return path;
}

void prof_log_line(const std::string &line)
{
   std::lock_guard<std::mutex> lk(g_prof_mtx);
   std::ofstream ofs(prof_log_path(), std::ios::app);
   ofs << line << '\n';
}

void prof_logf(const char *fmt, ...)
{
   char buf[1024];
   va_list ap;
   va_start(ap, fmt);
   std::vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   prof_log_line(buf);
}

ScopedTimerMs::ScopedTimerMs(const char *t, double *accumulate)
    : tag(t), t0(std::chrono::high_resolution_clock::now()), acc(accumulate)
{
}

ScopedTimerMs::~ScopedTimerMs()
{
   const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0)
                         .count();
   if (acc)
      *acc += ms;
   prof_logf("[PROF] %s %.3f", tag, ms);
}

} // namespace ANNS
