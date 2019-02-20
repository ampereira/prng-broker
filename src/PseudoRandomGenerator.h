#ifndef PseudoRandomGenerator_h
#define PseudoRandomGenerator_h

#ifndef __MIC__

	#include <cstdlib>
	#include <TRandom3.h>
	#include <random>
	#include <string>
	#include <boost/lexical_cast.hpp>
	#include <boost/thread.hpp>
	#include <boost/chrono.hpp>
	#include <boost/lockfree/queue.hpp>
	#include "pcg_random.hpp"
	#include "pcg_basic.h"
	#include "Timer.h"

#endif

#ifdef D_HEPF_INTEL
	#include <mkl.h>
	#include <mkl_vsl.h>
	#include <malloc.h>
	#include <stdio.h>
#endif

#ifdef D_GPU
	#include <cuda.h>
	#include <curand.h>

	#include <pthread.h>
	#include <sched.h>

	#define CUDA_CALL(x) if((x)!=cudaSuccess) { printf("Error at %s:%d code %d\n",__FILE__,__LINE__,x); return;}
	#define CURAND_CALL(x) if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d code %d\n",__FILE__,__LINE__,x); return;}
#endif

#define BUFFER_RATIO 0.01
#define MAX_PRNS 50000
// #define MAX_PRNS 100000



#ifndef __MIC__


enum Distribution {uniform, gaussian};
enum PRNG {MKL, MKLA1, MKLA2, MKLA3, TRandom, PCG, CURAND, KNC};



using namespace std;

// only works for doubles atm
class PseudoRandomGenerator {
	bool which_mkl = true;
	double *mkl_prns1, *mkl_prns2;
	double p1, p2;
    unsigned current_prn = 0;
	bool generated_mkl = true;
	bool generate_mkl = false;
    bool shutdown_prn = false;

	bool last_update = false;

	bool first_generated = false;

	TRandom3 *t_rnd;	// one per thread, Mersenne Twister
	pcg64_fast rnd; //(pcg_extras::seed_seq_from<std::random_device>{});
	pcg64_k32_fast rnd64;

	#ifdef D_HEPF_INTEL
	VSLStreamStatePtr mkl_stream;
	#endif

	PRNG prng_to_use = PRNG::TRandom;

	unsigned number_of_threads;
	unsigned number_of_consumer_threads;
	boost::thread *observer_thread = NULL;
	boost::thread *consumer_thread = NULL;
	boost::thread mkl_producer;
	boost::mutex get_mkl_prn, wait_mkl_mt;	// find better solution. atomic?
	boost::condition_variable wait_mkl;

	#ifdef D_KNC
	boost::mutex *wait_prns_mt, *wait_knc_produce_mt;
	boost::condition_variable *wait_prns, *wait_knc_produce;
	bool *which_knc_buffer;
	bool *next_knc_buffer;
	int *current_prn_knc, *current_prn_knc2;
	bool *first_generated_knc;
	bool *last_update_knc;
	VSLStreamStatePtr *knc_streams;
	double **knc_prns1, **knc_prns2;
	#endif

	normal_distribution<> normal;

	int current_mkl_prn = 0;
	int *current_mkl_prns;	// for mkla2
	bool last_gpu_update = false;		// not the same as which_mkl initial value

	double **gpu_prns1, **gpu_prns2;


	#ifdef D_GPU
	cudaStream_t streams[32];
	boost::condition_variable *gpu_wait_prn_request;	// one per thread
	unsigned *current_gpu_prn;	// for mkla2
	boost::mutex *wait_gpu_mt, *wait_for_gpu_prns;
	boost::condition_variable *awake_gpu;
	bool *which_gpu_buffer;
	bool *generated_gpu;
	boost::condition_variable *consumer_gpu_wait_prn;

	timeval t[50];
	long long unsigned final_time[50];
	#endif

	void observe1 (unsigned td);
	void observe2 (unsigned td); 
	void observe3 (unsigned td);
	// for profiling purposes
	double consume (unsigned td, unsigned mode);
	boost::mutex atomic_prn;




    bool is_lfq1 = true;
    unsigned *current_lfq_prn;

    // alternative simple arrays
    double _q1 [MAX_PRNS];
    double _q2 [MAX_PRNS];
    bool is_q1 = true;




    double prod = 0;

public:
	// param1 and param2 are the limits in uniform dist or mean and limit for gaussian
	PseudoRandomGenerator (void);
	void init (unsigned num_threads);
	~PseudoRandomGenerator (void);
	void initialize (double param1, double param2);
	void initialize (double param1, double param2, double seed);
	double uniformTrand (void);
	double uniform (void);
	double uniformPCG (void);
	double uniformMKL (void);
	void uniform2 (double *array, unsigned size);
	double gaussianTrand (void);
	double gaussian (void);
	double gauss (unsigned tid);
	double gaussian2 (void);
	double gaussianMKL (void);
	double gaussianMKLA (void);
	double gaussianMKLA3 (void);
	double gaussianMKLA2 (unsigned tid);
	double* gaussianMKLArray (int size);
	double gaussianGPU (void);
	double gaussianGPU2 (unsigned tid);
	void MKLArrayProducer (void);
	void GPUArrayProducer (void);
	void GPUArrayProducer2 (unsigned tid);
	void startGPU (void);
	void MKLArrayProducerLockFree (void);
	void shutdownMKL (void);
	void shutdownKNC (unsigned tid);
	void gaussian3 (double *array, unsigned size);
	void run (unsigned nthreads, unsigned mode);
	void stop (void);
	// for profiling purposes
	void runConsumers (unsigned nthreads, unsigned mode);
	void stopConsumers (void);
	void shutdownGPU (void);
	void reportGPU (void);

	void setGenerator(PRNG prng);
	double gaussianMKLKNC (unsigned tid);
public:
	void MKLArrayProducerKNC(unsigned tid);
};

#endif
#endif
