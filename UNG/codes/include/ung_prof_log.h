#ifndef UNG_PROF_LOG_H
#define UNG_PROF_LOG_H

#include <chrono>
#include <string>

namespace ANNS
{

const std::string &prof_log_path();
void prof_log_line(const std::string &line);
void prof_logf(const char *fmt, ...);

struct ScopedTimerMs
{
   const char *tag;
   std::chrono::high_resolution_clock::time_point t0;
   double *acc;

   explicit ScopedTimerMs(const char *t, double *accumulate = nullptr);
   ~ScopedTimerMs();
};

} // namespace ANNS

#endif // UNG_PROF_LOG_H
