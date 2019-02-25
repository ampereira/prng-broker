#include "Timer.h"

extern long unsigned event_counter;

using namespace std;

Timer::Timer (void) {
	average = 0;
	measuring = false;
	statistics.push_back(Statistic::Minimum);
}

Timer::Timer (vector<Statistic> _statistics) {
	average = 0;
	measuring = false;

	// avoids duplicates
	for (auto s1 : _statistics) {
		bool insert = true;

		for (auto s2 : statistics) {
			if (s1 == s2)
				insert = false;
		}

		if (insert)
			statistics.push_back(s1);
	}
}

Timer::Timer (Statistic _statistic) {
	average = 0;
	measuring = false;
	statistics.push_back(_statistic);
}

bool Timer::addStatistic (Statistic stat) {
	bool insert = true;

	for (auto s1 : statistics) {
		if (stat == s1)
			insert = false;
	}

	if (insert)
		statistics.push_back(stat);

	return insert;
}

/*inline
void Timer::errorReport (Error error, const char* file, const char* func) {
	string err;

	switch (as_integer(error)) {
		case 10 :
			err = "Statistic";
			break;
		case 11 :
			err = "Measurement";
			break;
		default :
			err = "Unknown error";
			break;
	}

	cerr << err << " in " << file << ":" << func << endl;
}*/

long long unsigned Timer::getAverage (void) {
	long long unsigned sum = 0; 

	for (auto &val : measurements)
		sum += val;

	average =  sum / measurements.size();

//	if (!average)
//		errorReport(Error::Statistic, __FILE__, __func__);

	return average;
}

long long unsigned Timer::getMinimum (void) {
	long long unsigned min = 0;

	for (auto meas : measurements)
		if (meas < min || min == 0)
			min = meas;

//	if (!min)
//		errorReport(Error::Statistic, __FILE__, __func__);

	return min;
}

long long unsigned Timer::getMaximum (void) {
	long long unsigned max = 0;

	for (auto meas : measurements)
		if (meas > max)
			max = meas;

//	if (!max)
//		errorReport(Error::Statistic, __FILE__, __func__);

	return max;
}

long long unsigned Timer::getStdDev (void) {

	if (!average)
		getAverage();

	long long unsigned accum = 0.0;
	std::for_each (measurements.begin(), measurements.end(), [&](const long long unsigned d) {
		accum += (d - average) * (d - average);
	});

	long long unsigned stdev = sqrt(accum / (measurements.size() - 1));

	return stdev;
}

void Timer::start (void) {
	gettimeofday(&t, NULL);
	initial_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}


long long unsigned Timer::stop (void) {
	gettimeofday(&t, NULL);
	long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;

	measurements.push_back(final_time - initial_time);

	return measurements.back();
}

long long unsigned Timer::getStatistic (Statistic stat) {
	long long unsigned result = 0;

	switch (as_integer(stat)) {
		case 0 :
			result = getAverage(); break;
		case 1 :
			result = getMinimum(); break;
		case 2 :
			result = getMaximum(); break;
		case 3 :
//			if (measurements.size() == 1)
//				errorReport(Error::Statistic, __FILE__, __func__);
//			else
				result = getStdDev();
			break;
//		default :
//			errorReport(Error::Statistic, __FILE__, __func__); break;
	}

	return result;
}

void Timer::reportVerbose (void) {
	int process_id;

	#ifdef D_MPI
	MPI_Comm_rank (MPI_COMM_WORLD, &process_id);
	#else
	process_id = 0;
	#endif

	if (process_id == 0) {
		cout << endl;
		cout << "**********************************" << endl;
		cout << "******* Measurement Report *******" << endl;
		cout << "**********************************" << endl;
		cout << endl;
	}

	for (auto stat : statistics) {
		switch (as_integer(stat)) {
			case 0 : 
				if (process_id == 0) cout << " => Average: ";
				break;
			case 1 : 
				if (process_id == 0) cout << " => Minimum: ";
				break;
			case 2 : 
				if (process_id == 0) cout << " => Maximum: ";
				break;
			case 3 : 
				if (process_id == 0) cout << " => StdDev: ";
				break;
		}

		#ifdef D_MPI
		long unsigned global_events;

		// bug in here...
		// MPI_Reduce(&event_counter, &global_events, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		int procs;
		MPI_Comm_size(MPI_COMM_WORLD, &procs);
		global_events = procs * event_counter;

		if (process_id == 0) {
			cout << results[stat]/1000000.f << " sec" << endl;
			cout << " => Event throughput: " << ((float)global_events / (results[stat]/1000000.f)) << " events/sec" << endl;
		}
		#else
		cout << results[stat]/1000000.f << " sec" << endl;
		cout << " => Event throughput: " << ((float)event_counter / (results[stat]/1000000.f)) << " events/sec" << endl;
		#endif
	}
}

// To be finished
map<Statistic,long long unsigned> Timer::report (Report type) {
	for (auto stat : statistics)
		results[stat] = getStatistic(stat);

	switch (as_integer(type)) {
		case 20 : reportVerbose(); break;
		case 21 : break;
	}

	return results;
}
