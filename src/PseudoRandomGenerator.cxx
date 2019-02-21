#include "PseudoRandomGenerator.h"


using namespace std;

#ifndef __MIC__
extern map<string, unsigned> thread_ids;

// // alternative lockfree
boost::lockfree::queue<double> lfq1 (MAX_PRNS);
boost::lockfree::queue<double> lfq2 (MAX_PRNS);
// bool is_lfq1 = true;
// unsigned *current_lfq_prn;

// // alternative simple arrays
// double _q1 [MAX_PRNS];
// double _q2 [MAX_PRNS];
// bool is_q1 = true;
// unsigned current_prn = 0;

// bool shutdown_prn = false;
#endif

#ifdef D_MKL
#ifdef D_KNC
void CheckVslError(int num) {
    switch(num) {
        case VSL_ERROR_CPU_NOT_SUPPORTED: {
            printf("Error: CPU version is not supported. (code %d).\n",num);
            break;
        }
        case VSL_ERROR_FEATURE_NOT_IMPLEMENTED: {
            printf("Error: this feature not implemented yet. (code %d).\n",num);
            break;
        }
        case VSL_ERROR_UNKNOWN: {
            printf("Error: unknown error (code %d).\n",num);
            break;
        }
        case VSL_ERROR_BADARGS: {
            printf("Error: bad arguments (code %d).\n",num);
            break;
        }
        case VSL_ERROR_MEM_FAILURE: {
            printf("Error: memory failure. Memory allocation problem maybe (code %d).\n",num);
            break;
        }
        case VSL_ERROR_NULL_PTR: {
            printf("Error: null pointer (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_INVALID_BRNG_INDEX: {
            printf("Error: invalid BRNG index (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED: {
            printf("Error: leapfrog initialization is unsupported (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED: {
            printf("Error: skipahead initialization is unsupported (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BRNGS_INCOMPATIBLE: {
            printf("Error: BRNGs are not compatible for the operation (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_STREAM: {
            printf("Error: random stream is invalid (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BRNG_TABLE_FULL: {
            printf("Error: table of registered BRNGs is full (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_STREAM_STATE_SIZE: {
            printf("Error: value in StreamStateSize field is bad (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_WORD_SIZE: {
            printf("Error: value in WordSize field is bad (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_NSEEDS: {
            printf("Error: value in NSeeds field is bad (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_NBITS: {
            printf("Error: value in NBits field is bad (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_UPDATE: {
            printf("Error: number of updated entries in buffer is invalid (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_NO_NUMBERS: {
            printf("Error: zero number of updated entries in buffer (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_INVALID_ABSTRACT_STREAM: {
            printf("Error: abstract random stream is invalid (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_FILE_CLOSE: {
            printf("Error: can`t close file (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_FILE_OPEN: {
            printf("Error: can`t open file (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_FILE_WRITE: {
            printf("Error: can`t write to file (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_FILE_READ: {
            printf("Error: can`t read from file (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_BAD_FILE_FORMAT: {
            printf("Error: file format is unknown (code %d).\n",num);
            break;
        }
        case VSL_RNG_ERROR_UNSUPPORTED_FILE_VER: {
            printf("Error: unsupported file version (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_ALLOCATION_FAILURE: {
            printf("Error: memory allocation failure in summary statistics functionality (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_DIMEN: {
            printf("Error: bad dimension value (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_OBSERV_N: {
            printf("Error: bad number of observations (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_STORAGE_NOT_SUPPORTED: {
            printf("Error: storage format is not supported (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_INDC_ADDR: {
            printf("Error: array of indices is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_WEIGHTS: {
            printf("Error: array of weights contains negative values (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MEAN_ADDR: {
            printf("Error: array of means is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_2R_MOM_ADDR: {
            printf("Error: array of 2nd order raw moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_3R_MOM_ADDR: {
            printf("Error: array of 3rd order raw moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_4R_MOM_ADDR: {
            printf("Error: array of 4th order raw moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_2C_MOM_ADDR: {
            printf("Error: array of 2nd order central moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_3C_MOM_ADDR: {
            printf("Error: array of 3rd order central moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_4C_MOM_ADDR: {
            printf("Error: array of 4th order central moments is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_KURTOSIS_ADDR: {
            printf("Error: array of kurtosis values is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_SKEWNESS_ADDR: {
            printf("Error: array of skewness values is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MIN_ADDR: {
            printf("Error: array of minimum values is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MAX_ADDR: {
            printf("Error: array of maximum values is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_VARIATION_ADDR: {
            printf("Error: array of variation coefficients is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_COV_ADDR: {
            printf("Error: covariance matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_COR_ADDR: {
            printf("Error: correlation matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_QUANT_ORDER_ADDR: {
            printf("Error: array of quantile orders is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_QUANT_ORDER: {
            printf("Error: bad value of quantile order (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_QUANT_ADDR: {
            printf("Error: array of quantiles is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_ORDER_STATS_ADDR: {
            printf("Error: array of order statistics is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_MOMORDER_NOT_SUPPORTED: {
            printf("Error: moment of requested order is not supported (code %d).\n",num);
            break;
        }
        case VSL_SS_NOT_FULL_RANK_MATRIX: {
            printf("Warning: correlation matrix is not of full rank (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_ALL_OBSERVS_OUTLIERS: {
            printf("Error: all observations are outliers (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_ROBUST_COV_ADDR: {
            printf("Error: robust covariance matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_ROBUST_MEAN_ADDR: {
            printf("Error: array of robust means is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_METHOD_NOT_SUPPORTED: {
            printf("Error: requested method is not supported (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_NULL_TASK_DESCRIPTOR: {
            printf("Error: task descriptor is null (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_OBSERV_ADDR: {
            printf("Error: dataset matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_SINGULAR_COV: {
            printf("Error: covariance matrix is singular (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_POOLED_COV_ADDR: {
            printf("Error: pooled covariance matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_POOLED_MEAN_ADDR: {
            printf("Error: array of pooled means is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_GROUP_COV_ADDR: {
            printf("Error: group covariance matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_GROUP_MEAN_ADDR: {
            printf("Error: array of group means is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_GROUP_INDC_ADDR: {
            printf("Error: array of group indices is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_GROUP_INDC: {
            printf("Error: group indices have improper values (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_ADDR: {
            printf("Error: array of parameters for outliers detection algorithm is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_N_ADDR: {
            printf("Error: pointer to size of parameter array for outlier detection algorithm is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_OUTLIERS_WEIGHTS_ADDR: {
            printf("Error: output of the outlier detection algorithm is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_ADDR: {
            printf("Error: array of parameters of robust covariance estimation algorithm is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_N_ADDR: {
            printf("Error: pointer to number of parameters of algorithm for robust covariance is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STORAGE_ADDR: {
            printf("Error: pointer to variable that holds storage format is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_PARTIAL_COV_IDX_ADDR: {
            printf("Error: array that encodes sub-components of random vector for partial covariance algorithm is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_PARTIAL_COV_ADDR: {
            printf("Error: partial covariance matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_PARTIAL_COR_ADDR: {
            printf("Error: partial correlation matrix is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_PARAMS_ADDR: {
            printf("Error: array of parameters for Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_PARAMS_N_ADDR: {
            printf("Error: pointer to number of parameters for Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_BAD_PARAMS_N: {
            printf("Error: bad size of the parameter array of Multiple Imputation method (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_PARAMS: {
            printf("Error: bad parameters of Multiple Imputation method (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_N_ADDR: {
            printf("Error: pointer to number of initial estimates in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_ADDR: {
            printf("Error: array of initial estimates for Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR: {
            printf("Error: array of simulated missing values in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N_ADDR: {
            printf("Error: pointer to size of the array of simulated missing values in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_ESTIMATES_N_ADDR: {
            printf("Error: pointer to the number of parameter estimates in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_ESTIMATES_ADDR: {
            printf("Error: array of parameter estimates in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N: {
            printf("Error: bad size of the array of simulated values in Multiple Imputation method (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_ESTIMATES_N: {
            printf("Error: bad size of array to hold parameter estimates obtained using Multiple Imputation method (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_OUTPUT_PARAMS: {
            printf("Error: array of output parameters in Multiple Imputation method is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_PRIOR_N_ADDR: {
            printf("Error: pointer to the number of prior parameters is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_PRIOR_ADDR: {
            printf("Error: array of prior parameters is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_MI_MISSING_VALS_N: {
            printf("Error: bad number of missing values (code %d).\n",num);
            break;
        }
        case VSL_SS_SEMIDEFINITE_COR: {
            printf("Warning: correlation matrix passed into parametrization function is semidefinite(code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_PARAMTR_COR_ADDR: {
            printf("Error: correlation matrix to be parametrized is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_COR: {
            printf("Error: all eigenvalues of correlation matrix to be parametrized are non-positive (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N_ADDR: {
            printf("Error: pointer to the number of parameters for quantile computation algorithm for streaming data is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_ADDR: {
            printf("Error: array of parameters of quantile computation algorithm for streaming data is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N: {
            printf("Error: bad number of parameters of quantile computation algorithm for streaming data (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS: {
            printf("Error: bad parameters of quantile computation algorithm for streaming data (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER_ADDR: {
            printf("Error: array of quantile orders for streaming data is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER: {
            printf("Error: bad quantile order for streaming data (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_STREAM_QUANT_ADDR: {
            printf("Error: array of quantiles for streaming data is not defined (code %d).\n",num);
            break;
        }
        case VSL_SS_ERROR_BAD_PARTIAL_COV_IDX: {
            printf("Error: partial covariance indices have improper values (code %d).\n",num);
            break;
        }
    }

    if(num < 0) {
       exit(1);
    }
}
// mem leak knc_prns

double* produceKNC (int brng, int seed, double p1, double p2) {
	
	int method = VSL_RNG_METHOD_GAUSSIAN_BOXMULLER;
	int errcode1=-1, errcode2=-1, errcode3=-1;

    double *knc_prn_buffer = new double [MAX_PRNS];
    VSLStreamStatePtr stream2;


	#pragma offload target(mic) in(brng, seed, stream2, method, p1, p2) out(knc_prn_buffer: length(MAX_PRNS))
	{
		/***** Initialize *****/
		errcode1 = vslNewStream( &stream2, brng, seed );
		//    CheckVslError( errcode );

		/***** Call RNG *****/
		errcode2 = vdRngGaussian( method, stream2, MAX_PRNS, knc_prn_buffer, p1, p2 );
		//   CheckVslError( errcode );

		/***** Deinitialize *****/
		errcode3 = vslDeleteStream( &stream2 );
		//    CheckVslError( errcode );
	}

	CheckVslError(errcode1);
	CheckVslError(errcode2);
	CheckVslError(errcode3);

	return knc_prn_buffer;
}
#endif
#endif

#ifndef __MIC__
void PseudoRandomGenerator::MKLArrayProducerKNC (unsigned tid) {
	#ifdef D_MKL
	#ifdef D_KNC
	int brng = VSL_BRNG_MT19937;
	int errcode;
	int nn = MAX_PRNS;
	int seed;

	seed = time(NULL);

	while (!shutdown_prn) {

		// if (last_update_knc[tid] != next_knc_buffer[tid]) {
			if (last_update_knc[tid] == false){

				// cout << "a criar 0 " << tid << endl;
				// delete knc_prns1[tid];

				knc_prns2[tid] = produceKNC(brng, seed, p1, p2);

				next_knc_buffer[tid] = true;

				// cout << "criou 0 " << tid << endl;

				first_generated_knc[tid] = true;
			} else {

				// cout << "a criar 1 " << tid << endl;
				// delete knc_prns2[tid];
				
				knc_prns1[tid] = produceKNC(brng, seed, p1, p2);
				next_knc_buffer[tid] = false;
				// cout << "criou 1 " << tid << endl;

				first_generated_knc[tid] = true;
			}

			// generated_mkl = true;
			// if (first_generated_knc[tid])
			// 	current_prn_knc[tid] = 0;

			{
				boost::unique_lock<boost::mutex> _lock (wait_prns_mt[tid]);
				wait_prns[tid].notify_all();
			}
			// boost::this_thread::sleep_for( boost::chrono::microseconds(50) );
			
		// }
		// {
		// 		boost::unique_lock<boost::mutex> _lock (wait_prns_mt[tid]);
		// 		wait_prns[tid].notify_all();
		// 	}
		{
            // cout << "bloqueou "<< endl;
			boost::unique_lock<boost::mutex> _lock (wait_knc_produce_mt[tid]);
			wait_knc_produce[tid].wait(_lock);
		}
		// cout << "chegou 2"<< endl;
	}
	#endif
	#endif
}

double PseudoRandomGenerator::gaussianMKLKNC (unsigned tid) {

	double val;

	#ifdef D_MKL
	#ifdef D_KNC

	if ((first_generated_knc[tid] == false || current_prn_knc2[tid] >= (MAX_PRNS * 0.99)) && which_knc_buffer[tid] == next_knc_buffer[tid]) {
		Timer ttt;
		long long unsigned stops;
		// cout << "ANTES "<< tid << " - "<<current_prn_knc2[tid] << " - " << which_knc_buffer[tid] << endl;
		ttt.start();
            wait_knc_produce[tid].notify_all();
		{
		boost::unique_lock<boost::mutex> _lock (wait_prns_mt[tid]);
		wait_prns[tid].wait(_lock);
		}
		stops = ttt.stop();
		// boost::this_thread::sleep_for( boost::chrono::microseconds(25) );
		// cout << "DEPOIS "<< tid << " - " << stops << endl;
	}

	if (which_knc_buffer[tid] == true) {
		// get_mkl_prn.lock();
	 	// cout << "DEPOIS" << current_prn_knc2[tid] << endl;
		val = knc_prns2[tid][current_prn_knc2[tid]++];
		// current_prn_knc[tid]++;

		if ((current_prn_knc2[tid] > (MAX_PRNS * 0.01)) && last_update_knc[tid] != which_knc_buffer[tid] ){
			// cout << "comeca a fazer o 1" << endl;
			
			// boost::unique_lock<boost::mutex> _lock (wait_knc_produce_mt[tid]);
			wait_knc_produce[tid].notify_all();
			last_update_knc[tid] = which_knc_buffer[tid];
		}

		if (current_prn_knc2[tid] == MAX_PRNS) {	
			which_knc_buffer[tid] = 1 - which_knc_buffer[tid];
			current_prn_knc2[tid] = 0;
		}

		// get_mkl_prn.unlock();
	} else {
		// get_mkl_prn.lock();
	 	// cout << "DEPOIS " << current_prn_knc2[tid] << endl;
		val = knc_prns1[tid][current_prn_knc2[tid]++];
		// current_prn_knc[tid]++;
	// cout << which_mkl << ":\t" << current_mkl_prn << endl;

		if ((current_prn_knc2[tid] > (MAX_PRNS * 0.01)) && last_update_knc[tid] != which_knc_buffer[tid] ) {
			// cout << "comeca a fazer o 0" << endl;
			
			boost::unique_lock<boost::mutex> _lock (wait_knc_produce_mt[tid]);
			wait_knc_produce[tid].notify_all();
			last_update_knc[tid] = which_knc_buffer[tid];
		}

		if (current_prn_knc2[tid] == MAX_PRNS) {	
			which_knc_buffer[tid] = 1 - which_knc_buffer[tid];
			current_prn_knc2[tid] = 0;
		}

		// get_mkl_prn.unlock();
	}
	#endif
	#endif

	return val;
}

// double prod = 0;
PseudoRandomGenerator::PseudoRandomGenerator (void) {

}

void PseudoRandomGenerator::init (unsigned num_threads) {
	number_of_threads = num_threads;

	t_rnd = new TRandom3 [number_of_threads];
	current_mkl_prns = new int [number_of_threads];

	int c = ((double)MAX_PRNS/(double)number_of_threads);
	// cout << "chegou " << c << "\t" << number_of_threads << endl;

	// for (unsigned i = 0; i < number_of_threads; i++) {
	// 	current_mkl_prns[i] = i * c;
	// 	// cout << "Thread " << i << ":\t" << current_mkl_prns[i] << endl;
	// }

	mkl_prns1 = new double [MAX_PRNS];
	mkl_prns2 = new double [MAX_PRNS];

	#ifdef D_KNC
	wait_prns_mt = new boost::mutex [number_of_threads];
	wait_prns = new boost::condition_variable [number_of_threads];
	wait_knc_produce_mt = new boost::mutex [number_of_threads];
	wait_knc_produce = new boost::condition_variable [number_of_threads];

	current_prn_knc = new int [number_of_threads];
	current_prn_knc2 = new int [number_of_threads];
	which_knc_buffer = new bool [number_of_threads];
	next_knc_buffer = new bool [number_of_threads];
	first_generated_knc = new bool [number_of_threads];
	last_update_knc = new bool [number_of_threads];

    knc_streams = new VSLStreamStatePtr[number_of_threads];

    knc_prns1 = new double*[number_of_threads];
    knc_prns2 = new double*[number_of_threads];

	for (unsigned i = 0; i < number_of_threads; i++) {
		which_knc_buffer[i] = true;
		next_knc_buffer[i] = true;
		first_generated_knc[i] = false;
		current_prn_knc2[i] = 0;
		last_update_knc[i] = false;
	}
	#endif

	#ifdef D_GPU
	gpu_prns1 = new double* [number_of_threads];
	gpu_prns2 = new double* [number_of_threads];

	// streams = new cudaStream_t [number_of_threads];

	gpu_wait_prn_request = new boost::condition_variable [number_of_threads];
	consumer_gpu_wait_prn = new boost::condition_variable [number_of_threads];

	current_gpu_prn = new unsigned [number_of_threads];

	wait_gpu_mt = new boost::mutex [number_of_threads];
	wait_for_gpu_prns = new boost::mutex [number_of_threads];



	awake_gpu = new boost::condition_variable [number_of_threads];

	which_gpu_buffer = new bool[number_of_threads];
	generated_gpu = new bool[number_of_threads];

	for (unsigned i = 0; i < number_of_threads; i++) {

		// #ifdef D_MKL
		// gpu_prns1[i] = (double*) _mm_malloc (MAX_PRNS * sizeof(double), 64);
		// gpu_prns2[i] = (double*) _mm_malloc (MAX_PRNS * sizeof(double), 64);
		// #else
		// gpu_prns1[i] = new double[MAX_PRNS];
		// gpu_prns2[i] = new double[MAX_PRNS];
		// #endif
		// cudaMallocHost(&gpu_prns1[i], MAX_PRNS * sizeof(double));
		// cudaMallocHost(&gpu_prns2[i], MAX_PRNS * sizeof(double));

		current_gpu_prn[i] = 0;

		final_time[i] = 0;

		which_gpu_buffer[i] = true;
		generated_gpu[i] = false;

	}
	#endif
}

void PseudoRandomGenerator::initialize (double param1, double param2) {

	p1 = param1;
	p2 = param2;

	normal_distribution<> d (p1,p2);
	normal = d;

	#ifdef D_MKL
	vslNewStream( &mkl_stream, VSL_BRNG_MT19937, time(NULL) );
	#endif
	
}

void PseudoRandomGenerator::initialize (double param1, double param2, double seed) {

	p1 = param1;
	p2 = param2;

	normal_distribution<> d (p1,p2);
	normal = d;

	for (unsigned i = 0; i < number_of_threads; i++)
		t_rnd[i].SetSeed(seed);

	#ifdef D_MKL
	vslNewStream( &mkl_stream, VSL_BRNG_MT19937, seed );
	#endif
}

PseudoRandomGenerator::~PseudoRandomGenerator (void) {
	#ifdef D_MKL
	vslDeleteStream( &mkl_stream );
	#endif

	shutdown_prn = true;
	{
		boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
		wait_mkl.notify_all();
	}
	delete mkl_prns1;
	delete mkl_prns2;
}

double PseudoRandomGenerator::uniformTrand (void) {
	string threadId = boost::lexical_cast<std::string>(boost::this_thread::get_id());
	unsigned _this_thread_id = thread_ids.find(threadId)->second;
	#ifdef D_SEQ
	_this_thread_id = 0;
	#endif

	return t_rnd[_this_thread_id].Rndm();
}

double PseudoRandomGenerator::uniformMKL (void) {
	double val;
	
	#ifdef D_MKL
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, mkl_stream, 1, &val, 0.0, 1.0);
	#else
	val = 0.0;
	#endif

	return val;
}

double PseudoRandomGenerator::uniformPCG (void) {
	return (double)rnd()/18446744073709551615.0;
}


double PseudoRandomGenerator::uniform (void) {

    switch (prng_to_use) {
        case PCG: return uniformPCG(); break;

        #ifdef D_MKL
        case MKL: return uniformMKL(); break;
        case MKLA1: return uniformMKL(); break;
        case MKLA2: return uniformMKL(tid); break;
        case MKLA3: return uniformMKL(); break;  
        #elif D_ROOT
        case TRandom: return uniformTrand(); break;
        #elif D_GPU
        case CURAND: cerr << "Not supported yet" << endl; exit(0); break;
        #elif D_KNC
        case KNC: cerr << "Not supported yet" << endl; exit(0); break;
        #endif
        default: cerr << "PRNG not supported - check lib compilation options are set correctly" << endl;
                 exit(0); break;
    }
}

void PseudoRandomGenerator::uniform2 (double *array, unsigned size) {
	for (size_t i = 0; i < size; i++) {
		array[i] = (double)rnd()/18446744073709551615.0;
	}
}

double PseudoRandomGenerator::gaussianTrand (void) {
	string threadId = boost::lexical_cast<std::string>(boost::this_thread::get_id());
	unsigned _this_thread_id = thread_ids.find(threadId)->second;
	#ifdef D_SEQ
	_this_thread_id = 0;
	#endif

	return t_rnd[_this_thread_id].Gaus(p1, p2);
}

double* PseudoRandomGenerator::gaussianMKLArray (int size) {
	double *val;
	val = new double[size];
	
	#ifdef D_MKL
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mkl_stream, size, val, p1, p2);
	#else
	val = NULL;
	#endif

	return val;
}

double PseudoRandomGenerator::gaussianMKL (void) {
	double val;
	
	#ifdef D_MKL
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mkl_stream, 1, &val, p1, p2);
	#else
	val = 0.0;
	#endif

	return val;
}

void PseudoRandomGenerator::shutdownMKL (void) {
	shutdown_prn = true;
	{
		boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
		wait_mkl.notify_all();
	}
}


void PseudoRandomGenerator::shutdownKNC (unsigned tid) {
	#ifdef D_KNC
	shutdown_prn = true;
	{
		boost::unique_lock<boost::mutex> _lock (wait_knc_produce_mt[tid]);
		wait_knc_produce[tid].notify_all();
	}
	#endif
}

void PseudoRandomGenerator::MKLArrayProducer (void) {
	#ifdef D_MKL
	while (!shutdown_prn) {

		if (which_mkl == true){
			// generate_mkl = true;
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mkl_stream, MAX_PRNS, mkl_prns1, p1, p2);
			current_prn = 0;
			// cout << endl<<"GEROU PRNS1" << endl << endl;
		} else {
			// generate_mkl = true;
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mkl_stream, MAX_PRNS, mkl_prns2, p1, p2);
			current_prn = 0;
			// cout << endl<<"GEROU PRNS2" << endl << endl;
		}

		generated_mkl = true;

		{
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.wait(_lock);
		}
	}
	#endif
}

void PseudoRandomGenerator::MKLArrayProducerLockFree (void) {
	#ifdef D_MKL
	double val;

	while (!shutdown_prn) {
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, mkl_stream, 1, &val, p1, p2);
		lfq1.push(val);
		// cout << endl << "tau"<< endl << endl;
	}

	#endif
}

double PseudoRandomGenerator::gaussianMKLA3 (void) {
	double val;

	while (lfq1.empty() == true) {
		boost::this_thread::sleep_for( boost::chrono::microseconds(10) );
		// cout << endl << "pois"<< endl << endl;
	}

	lfq1.pop(val);

	return val;
}

// array positions not shared among threads
double PseudoRandomGenerator::gaussianMKLA2 (unsigned tid) {
	double val;
	bool last_update = false;		// not the same as which_mkl initial value

	#ifdef D_MKL
	if (which_mkl == true) {
		val = mkl_prns2[current_mkl_prns[tid]++];
	// cout << which_mkl << ":\t" << current_mkl_prn << endl;

		// if ((current_mkl_prns[tid] > (MAX_PRNS * 0.9)) && last_update != which_mkl ){
		if ((current_mkl_prns[tid] == (MAX_PRNS * 0.9)) /*&& !generate_mkl*/ ){
			// cout << tid << " PRNS2 " << current_mkl_prns[tid] << " " << (MAX_PRNS * 0.9) << " " << generate_mkl<< endl;
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
			last_update = which_mkl;
			generate_mkl = true;
			generated_mkl = false;
		}

		get_mkl_prn.lock();
		if (current_mkl_prns[tid] >= MAX_PRNS && generated_mkl) {
			generate_mkl = false;
			which_mkl = 1 - which_mkl;

			// cout << endl << "MUDOU para 1"<< endl << endl;
			// exit(0);

			int c = ((double)MAX_PRNS/(double)number_of_threads);
			for (unsigned i = 0; i < number_of_threads; i++)
				current_mkl_prns[i] = i * c;
		}

		get_mkl_prn.unlock();
	} else {
		val = mkl_prns1[current_mkl_prns[tid]++];
	// cout << which_mkl << ":\t" << current_mkl_prn << endl;

		// if ((current_mkl_prns[tid] > (MAX_PRNS * 0.9)) && last_update != which_mkl ) {
		if ((current_mkl_prns[tid] == (MAX_PRNS * 0.9)) /*&& !generate_mkl*/ ) {
			// cout << tid << " PRNS1 " << current_mkl_prns[tid] << " " << (MAX_PRNS * 0.9) << " " << generate_mkl<< endl;
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
			last_update = which_mkl;
		}

		get_mkl_prn.lock();
		if (current_mkl_prns[tid] == MAX_PRNS && generated_mkl) {	
			generate_mkl = false;
			which_mkl = 1 - which_mkl;

			// cout << endl << "MUDOU para 2"<< endl << endl;

			int c = ((double)MAX_PRNS/(double)number_of_threads);
			for (unsigned i = 0; i < number_of_threads; i++)
				current_mkl_prns[i] = i * c;
		}

		get_mkl_prn.unlock();
	}
	#endif

	return val;
}

// chamar mesmo pela thread
void PseudoRandomGenerator::GPUArrayProducer2 (unsigned tid) {
	#ifdef D_GPU

	#ifdef D_AFFINITY
	cpu_set_t cpuset;
	CPU_ZERO( & cpuset);
	CPU_SET( tid+boost::thread::physical_concurrency(), & cpuset);

	int erro(::pthread_setaffinity_np( ::pthread_self(), sizeof( cpuset), & cpuset));
	#endif


	size_t i;

	curandGenerator_t gens;
	double *devData;
	double *data;
	bool which_mkl_gen = true;

	/* Allocate n floats on device */
	CUDA_CALL(cudaMalloc((void **)&devData, MAX_PRNS*sizeof(double)));

	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gens, CURAND_RNG_PSEUDO_DEFAULT));
	CUDA_CALL(cudaStreamCreate(&streams[tid]));

	// set specific stream for this thread
	CURAND_CALL(curandSetStream(gens, streams[tid]));

	while (!shutdown_prn) {

		// CURAND_CALL(curandGenerateNormalDouble(gens[tid], devData, MAX_PRNS, p1, p2));
		CURAND_CALL(curandGenerateNormalDouble(gens, devData, MAX_PRNS, p1, p2));
		
		cudaMallocHost(&data, MAX_PRNS*sizeof(double));

		// cout << endl << endl << "CHEGOU1 "<< tid << endl << endl;

		if (which_mkl_gen == true){
			cudaFreeHost(gpu_prns1[tid]);
			/* Copy device memory to host */
			CUDA_CALL(cudaMemcpyAsync(data, devData, MAX_PRNS * sizeof(double), cudaMemcpyDeviceToHost, streams[tid]));
			

			CUDA_CALL(cudaStreamSynchronize(streams[tid]));

			gpu_prns1[tid] = data;

			// cout << endl << endl << "gerou prns1\t"<< tid << endl << endl;
			which_mkl_gen = false;
		} else {
			cudaFreeHost(gpu_prns2[tid]);
			/* Copy device memory to host */
			CUDA_CALL(cudaMemcpyAsync(data, devData, MAX_PRNS * sizeof(double), cudaMemcpyDeviceToHost, streams[tid]));
			

			CUDA_CALL(cudaStreamSynchronize(streams[tid]));

			gpu_prns2[tid] = data;

			// cout << endl << endl << "gerou prns2\t"<< tid << endl << endl;
			which_mkl_gen = true;
		}

		// cout << endl << endl << "CHEGOU3 "<< tid << endl << endl;

		generated_gpu[tid] = true;

		{
			boost::unique_lock<boost::mutex> _lock (wait_for_gpu_prns[tid]);
			consumer_gpu_wait_prn[tid].notify_all();
		}

		{
			boost::unique_lock<boost::mutex> _lock (wait_gpu_mt[tid]);
			gpu_wait_prn_request[tid].wait(_lock);
			generated_gpu[tid] = false;
		}
	}

	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gens));
	CUDA_CALL(cudaFree(devData));
	
	#endif
}

double PseudoRandomGenerator::gaussianGPU2 (unsigned tid) {
	double val;
	unsigned current;

	#ifdef D_GPU

	if (!generated_gpu[tid]) {
		{
			gettimeofday(&t[tid], NULL);
			long long unsigned initial_time = t[tid].tv_sec * TIME_RESOLUTION + t[tid].tv_usec;
			
			boost::unique_lock<boost::mutex> _lock (wait_for_gpu_prns[tid]);
			consumer_gpu_wait_prn[tid].wait(_lock);

			gettimeofday(&t[tid], NULL);
			final_time[tid] = t[tid].tv_sec * TIME_RESOLUTION + t[tid].tv_usec;
		}
	}

	if (current_gpu_prn[tid] == 0){
		// cout << "a acabar prn2\t" << tid << endl;

		boost::unique_lock<boost::mutex> _lock (wait_gpu_mt[tid]);
		gpu_wait_prn_request[tid].notify_all();
	}

	// cout << "chegou" << endl;

	if (which_gpu_buffer[tid] == true) {
		current_gpu_prn[tid]++;

		if (current_gpu_prn[tid] == MAX_PRNS-1) {	
			which_gpu_buffer[tid] = 1 - which_gpu_buffer[tid];
			current_gpu_prn[tid] = 0;

			// cout << "acabou prn1\t" << tid << endl;
		}


		val = gpu_prns1[tid][current];

	} else {
		current_gpu_prn[tid]++;


		if (current_gpu_prn[tid] == MAX_PRNS-1) {	
			which_gpu_buffer[tid] = 1 - which_gpu_buffer[tid];
			current_gpu_prn[tid] = 0;
			// cout << "acabou prn2\t" << tid << endl;
		}

		val = gpu_prns2[tid][current];
	}

	#endif

	return val;
}

void PseudoRandomGenerator::shutdownGPU (void) {
	#ifdef D_GPU
	shutdown_prn = true;

	for (unsigned i = 0; i < number_of_threads; i++) {
		boost::unique_lock<boost::mutex> _lock (wait_gpu_mt[i]);
		gpu_wait_prn_request[i].notify_all();
	}
	#endif
}

void PseudoRandomGenerator::reportGPU (void) {
	#ifdef D_GPU
	cout << "Thread\tPRNG wait time (us)"<<endl << "----------------------------------" << endl;
	for (unsigned i = 0; i < number_of_threads; i++) {
		cout << i << "\t" << final_time[i] << endl;
	}
	#endif
}

void PseudoRandomGenerator::GPUArrayProducer (void) {
	#ifdef D_GPU

	size_t i;

	curandGenerator_t gen;
	double *devData;
	bool which_mkl_gen = true;

	/* Allocate n floats on device */
	CUDA_CALL(cudaMalloc((void **)&devData, MAX_PRNS*sizeof(double)));

	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); 

	while (!shutdown_prn) {
		/* Generate n floats on device */
		// cudaEvent_t start, stop;
		// cudaEventCreate(&start);
		// cudaEventCreate(&stop);
		// cudaEventRecord(start);

		CURAND_CALL(curandGenerateNormalDouble(gen, devData, MAX_PRNS, p1, p2));

		if (which_mkl_gen == true){
			// generate_mkl = true;
			/* Copy device memory to host */
			CUDA_CALL(cudaMemcpy(mkl_prns1, devData, MAX_PRNS * sizeof(double), cudaMemcpyDeviceToHost));
			// cudaEventRecord(stop);
			// cudaEventSynchronize(stop);
			// float milliseconds = 0;
			// cudaEventElapsedTime(&milliseconds, start, stop);
			// cout << milliseconds << endl;
			// exit(0);
			// current_prn = 0;
			which_mkl_gen = false;
			// cout << endl<<"GEROU PRNS1" << endl << endl;
		} else {
			// generate_mkl = true;
			/* Copy device memory to host */
			CUDA_CALL(cudaMemcpy(mkl_prns2, devData, MAX_PRNS * sizeof(double), cudaMemcpyDeviceToHost));
			// current_prn = 0;
			which_mkl_gen = true;
			// cout << endl<<"GEROU PRNS2" << endl << endl;
		}

		generated_mkl = true;

		{
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.wait(_lock);
		}
	}
	// cout << "chegou"<< endl;

	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gen));
	CUDA_CALL(cudaFree(devData));
	
	#endif
}

double PseudoRandomGenerator::gaussianGPU (void) {
	double val;
	unsigned current;

	#ifdef D_GPU

	if (which_mkl == true) {
		get_mkl_prn.lock();
		current = current_mkl_prn++;
		
		if ((current_mkl_prn > (MAX_PRNS * 0.5)) && last_gpu_update != which_mkl ){
			last_gpu_update = true;
			// tt.start();

			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
			// cout << endl << "DESBLOQUEOU " << current_mkl_prn<<endl << endl;
		}

		if (current_mkl_prn == MAX_PRNS-1) {	
			which_mkl = 1 - which_mkl;
			current_mkl_prn = 0;
			// tt.stop();
			// tt.report(Report::Verbose);
			// exit(0);
			// cout << endl << "CONSUMIDO 1" << endl << endl;
		}
		get_mkl_prn.unlock();

		val = mkl_prns1[current];

	} else {
		get_mkl_prn.lock();
		current = current_mkl_prn++;
		
		if ((current_mkl_prn > (MAX_PRNS * 0.5)) && last_gpu_update != which_mkl ){
			last_gpu_update = false;
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
		}

		if (current_mkl_prn == MAX_PRNS-1) {	
			which_mkl = 1 - which_mkl;
			current_mkl_prn = 0;
			// cout << endl << "CONSUMIDO 2" << endl << endl;
		}
		get_mkl_prn.unlock();

		val = mkl_prns2[current];
	}

	#endif

	return val;
}

double PseudoRandomGenerator::gauss (unsigned tid) {

	switch (prng_to_use) {
        case PCG: return gaussian2(); break;

        #ifdef D_MKL
        case MKL: return gaussianMKL(); break;
        case MKLA1: return gaussianMKLA(); break;
        case MKLA2: return gaussianMKLA2(tid); break;
        case MKLA3: return gaussianMKLA3(); break;  
        #elif D_ROOT
		case TRandom: return gaussianTrand(); break;
        #elif D_GPU
        case CURAND: return gaussianGPU2(tid); break;
        #elif D_KNC
        case KNC: return gaussianMKLKNC(tid); break;
        #endif
        default: cerr << "PRNG not supported - check lib compilation options are set correctly" << endl;
                 exit(0); break;
	}
}

void PseudoRandomGenerator::setGenerator (PRNG prng) {
	prng_to_use = prng;
}

double PseudoRandomGenerator::gaussianMKLA (void) {
	double val;
	bool last_update = false;		// not the same as which_mkl initial value

	#ifdef D_MKL
	if (which_mkl == true) {
		get_mkl_prn.lock();
		val = mkl_prns2[current_mkl_prn++];
	// cout << which_mkl << ":\t" << current_mkl_prn << endl;

		if ((current_mkl_prn > (MAX_PRNS * 0.9)) && last_update != which_mkl ){
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
			last_update = which_mkl;
		}

		if (current_mkl_prn == MAX_PRNS) {	
			which_mkl = 1 - which_mkl;
			current_mkl_prn = 0;
		}

		get_mkl_prn.unlock();
	} else {
		get_mkl_prn.lock();
		val = mkl_prns1[current_mkl_prn++];
	// cout << which_mkl << ":\t" << current_mkl_prn << endl;

		if ((current_mkl_prn > (MAX_PRNS * 0.9)) && last_update != which_mkl ) {
			boost::unique_lock<boost::mutex> _lock (wait_mkl_mt);
			wait_mkl.notify_all();
			last_update = which_mkl;
		}

		if (current_mkl_prn == MAX_PRNS) {	
			which_mkl = 1 - which_mkl;
			current_mkl_prn = 0;
		}

		get_mkl_prn.unlock();
	}
	#endif

	return val;
}

double PseudoRandomGenerator::gaussian (void) {
	return normal(rnd);
}

// using box muller from ROOT
// better than gaussian3
double PseudoRandomGenerator::gaussian2 (void) {
	const double kC1 = 1.448242853;
   const double kC2 = 3.307147487;
   const double kC3 = 1.46754004;
   const double kD1 = 1.036467755;
   const double kD2 = 5.295844968;
   const double kD3 = 3.631288474;
   const double kHm = 0.483941449;
   const double kZm = 0.107981933;
   const double kHp = 4.132731354;
   const double kZp = 18.52161694;
   const double kPhln = 0.4515827053;
   const double kHm1 = 0.516058551;
   const double kHp1 = 3.132731354;
   const double kHzm = 0.375959516;
   const double kHzmp = 0.591923442;
   /*zhm 0.967882898*/

   const double kAs = 0.8853395638;
   const double kBs = 0.2452635696;
   const double kCs = 0.2770276848;
   const double kB  = 0.5029324303;
   const double kX0 = 0.4571828819;
   const double kYm = 0.187308492 ;
   const double kS  = 0.7270572718 ;
   const double kT  = 0.03895759111;

   double result;
   double rn,x,y,z;

   do {
      y = uniform();

      if (y>kHm1) {
         result = kHp*y-kHp1; break; }

      else if (y<kZm) {
         rn = kZp*y-1;
         result = (rn>0) ? (1+rn) : (-1+rn);
         break;
      }

      else if (y<kHm) {
         rn = uniform();
         rn = rn-1+rn;
         z = (rn>0) ? 2-rn : -2-rn;
         if ((kC1-y)*(kC3+abs(z))<kC2) {
            result = z; break; }
         else {
            x = rn*rn;
            if ((y+kD1)*(kD3+x)<kD2) {
               result = rn; break; }
            else if (kHzmp-y<exp(-(z*z+kPhln)/2)) {
               result = z; break; }
            else if (y+kHzm<exp(-(x+kPhln)/2)) {
               result = rn; break; }
         }
      }

      while (1) {
         x = uniform();
         y = kYm * uniform();
         z = kX0 - kS*x - y;
         if (z>0)
            rn = 2+y/x;
         else {
            x = 1-x;
            y = kYm-y;
            rn = -(2+y/x);
         }
         if ((y-kAs+x)*(kCs+x)+kBs<0) {
            result = rn; break; }
         else if (y<x+kT)
            if (rn*rn<4*(kB-log(x))) {
               result = rn; break; }
      }
   } while(0);

   return p1 + p2 * result;
}

// optimized? box only does 1 iteration of the do/while
void PseudoRandomGenerator::gaussian3 (double *array, unsigned size) {
	unsigned limit = size * 1.5;
	double prns [limit];
	unsigned prn = 0;

	const double kC1 = 1.448242853;
	const double kC2 = 3.307147487;
	const double kC3 = 1.46754004;
	const double kD1 = 1.036467755;
	const double kD2 = 5.295844968;
	const double kD3 = 3.631288474;
	const double kHm = 0.483941449;
	const double kZm = 0.107981933;
	const double kHp = 4.132731354;
	const double kZp = 18.52161694;
	const double kPhln = 0.4515827053;
	const double kHm1 = 0.516058551;
	const double kHp1 = 3.132731354;
	const double kHzm = 0.375959516;
	const double kHzmp = 0.591923442;
	/*zhm 0.967882898*/

	const double kAs = 0.8853395638;
	const double kBs = 0.2452635696;
	const double kCs = 0.2770276848;
	const double kB  = 0.5029324303;
	const double kX0 = 0.4571828819;
	const double kYm = 0.187308492 ;
	const double kS  = 0.7270572718 ;
	const double kT  = 0.03895759111;


	uniform2(prns, limit);

	for (unsigned i = 0; i < size; i++) {
			double result;
			double rn,x,y,z;
		 do {
	       y = prns[prn++];

	       if (y>kHm1) {
	          result = kHp*y-kHp1; break; }

	       else if (y<kZm) {
	          rn = kZp*y-1;
	          result = (rn>0) ? (1+rn) : (-1+rn);
	          break;
	       }

	       else if (y<kHm) {
	          rn = prns[prn++];
	          rn = rn-1+rn;
	          z = (rn>0) ? 2-rn : -2-rn;
	          if ((kC1-y)*(kC3+abs(z))<kC2) {
	             result = z; break; }
	          else {
	             x = rn*rn;
	             if ((y+kD1)*(kD3+x)<kD2) {
	                result = rn; break; }
	             else if (kHzmp-y<exp(-(z*z+kPhln)/2)) {
	                result = z; break; }
	             else if (y+kHzm<exp(-(x+kPhln)/2)) {
	                result = rn; break; }
	          }
	       }

	       while (1) {
	          x = prns[prn++];
	          y = kYm * prns[i * 3 + 2];
	          z = kX0 - kS*x - y;
	          if (z>0)
	             rn = 2+y/x;
	          else {
	             x = 1-x;
	             y = kYm-y;
	             rn = -(2+y/x);
	          }
	          if ((y-kAs+x)*(kCs+x)+kBs<0) {
	             result = rn; break; }
	          else if (y<x+kT)
	             if (rn*rn<4*(kB-log(x))) {
	                result = rn; break; }
	       }
	    } while(0);

		array[i] = p1 + p2 * result;
	}
}

#endif
