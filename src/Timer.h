#ifndef TIMER_h
#define TIMER_h

#include <sys/time.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#ifdef D_MPI
#include <mpi.h>
#endif

#define TIME_RESOLUTION 1000000	// time measuring resolution (us)

using namespace std;

enum class Statistic {
	Average=0,
	Minimum=1,
	Maximum=2,
	StdDev=3
};

enum class Error {
	Statistic=10,
	Measurement=11,
	OUT_OF_BOUNDS=12
};

enum class Report {
	Verbose=20,
	Csv=21
};

template <typename Enumeration>
auto as_integer(Enumeration const value)
	-> typename std::underlying_type<Enumeration>::type
{
	return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

class Timer {
protected:
	long long unsigned initial_time;	// initial time to get the end-init
	vector<long long unsigned> measurements;			// measurements performed
	bool measuring;						// true after start is called and false after stop is called
	timeval t;							// holds the data of gettimeofday
	vector<Statistic> statistics;  // holds the report statistics
	map<Statistic,long long unsigned> results;	// map to hold the Statistic-value pair
//	Error error;
	long long unsigned average;

	long long unsigned getStatistic (Statistic);
	long long unsigned getAverage (void);
	long long unsigned getMinimum (void);
	long long unsigned getMaximum (void);
	long long unsigned getStdDev (void);
//	void errorReport (Error, const char*, const char*);

public:
	void reportVerbose (void);
	Timer (void);
	Timer (std::vector<Statistic>);
	Timer (Statistic);
	void start (void);
	long long unsigned stop (void);
	map<Statistic,long long unsigned> report (Report);
	bool addStatistic (Statistic);
};

#endif
