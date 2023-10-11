/*
 *  nonmaxsup_stub_single.c
 
	Stub for manifold detection by non-maximum suppresion criteria, input data must be 
	one-dimensional arrays, input/output data are single precision float.
 
 
 *  
 *
 *  Created by Antonio Martínez on 1/22/13.
 *  Copyright 2013 Universidad Almeria. All rights reserved.
 *
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

// Constants
#define DBL_EPSILON 2.2204460492503131E-16
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
#define INTER_FACTOR .71
#define CACHE_USED .33
#define BUFFER_SIZE 1024
#define BYTE_PER_KBYTE 1024
#define ERROR_FILE_READ -1
#define ERROR_CACHE_NOT_FOUND -2
#define ERROR_MEM_NOT_FOUND -3


#define SQR(x)      ((x)*(x))
// Global Variables
static int sq, lq, ld, task_size;
//static long long int *M;
//static double *Ixx, *Iyy, *Izz, *Ixy, *Ixz, *Iyz;
//static double *L1, *L2, *L3;
//static double *I, *V1x, *V1y, *V1z, *V2x, *V2y, *V2z, *V3x, *V3y, *V3z;
//static unsigned char *F;
static pthread_mutex_t mutex;

// Global functions
static void* look_neigbourhood( void* ptr );
static int dsyevv3(double A[3][3], double Q[3][3], double w[3]);
static int dsyevc3(double A[3][3], double w[3]);
static void* desyevv3stub( void* ptr );
static long long int get_cache_size();
//static int get_M(double *mat[]);
//static int get_N(int *mat[]);

//static PyObject * supression_nonmaxsup(PyObject *self, PyObject *args);

//Funcion del modulo


typedef struct{
//inputs
    double* I;
	double* V1x;
	double* V1y;
	double* V1z;
	double* V2x;
	double* V2y;
	double* V2z;
	long long int* M;
	unsigned int* dim;

    //Output
	unsigned int* F ;
    }Tomos;
static PyObject * supression_nonmaxsup(PyObject *self, PyObject *args)
{	//Inputs
	PyObject* I_array;//double
	PyObject* V1x_array;//double
	PyObject* V1y_array;//double
	PyObject* V1z_array;//double
	PyObject* V2x_array;//double
	PyObject* V2y_array;//double
	PyObject* V2z_array;//double
	PyObject* M_array;//long long int
	PyObject* dim_array;//unsigned int
	


	//Auxiliar variables
	int i, nta, nth, num_threads;
	//int type;
	int m, mh;
	npy_intp mn, mhn;
	//size_t len, len64;
	long long int dat64;
	pthread_t* threads;
	//int sq, lq, ld, task_size;
	//pthread_mutex_t mutex;

	unsigned int* F ;
	


	
	if (!PyArg_ParseTuple(args,"OOOOOOOOO",&I_array,&V1x_array,&V1y_array,&V1z_array,&V2x_array,&V2y_array,&V2z_array,&M_array,&dim_array)){
	    printf("ERROR: nonmaxsup: Unable to load inputs.\n");
	    PyErr_SetString(PyExc_TypeError, "Unable to load inputs.\n");
		return NULL;}

	//Transform to NumPy matrix
	PyArrayObject* I_np_array = (PyArrayObject*)PyArray_FROM_OTF(I_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (I_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming I in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform I in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V1x_np_array = (PyArrayObject*)PyArray_FROM_OTF(V1x_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V1x_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V1x in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V1x in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V1y_np_array = (PyArrayObject*)PyArray_FROM_OTF(V1y_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V1y_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V1y in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V1y in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V1z_np_array = (PyArrayObject*)PyArray_FROM_OTF(V1z_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V1z_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V1z in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V1z in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V2x_np_array = (PyArrayObject*)PyArray_FROM_OTF(V2x_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V2x_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V2x in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V2x in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V2y_np_array = (PyArrayObject*)PyArray_FROM_OTF(V2y_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V2y_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V2y in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V2y in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* V2z_np_array = (PyArrayObject*)PyArray_FROM_OTF(V2z_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (V2y_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming V2z in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform V2z in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* M_np_array = (PyArrayObject*)PyArray_FROM_OTF(M_array, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	if (M_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming M in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform M in a NumPy Matrix.\n");
        return NULL;
    }

	PyArrayObject* dim_np_array = (PyArrayObject*)PyArray_FROM_OTF(dim_array, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
	if (dim_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming dim in a NumPy matrix.\n");
        printf("ERROR: nonmaxsup: Unable to transform dim in a NumPy Matrix.\n");
        return NULL;
    }


    //Checking dimensions

    mn= PyArray_DIMS(I_np_array)[0];
    m=(int)mn;

    if (PyArray_DIMS(V1x_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;

    }

    if (PyArray_DIMS(V1y_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;
    }

    if (PyArray_DIMS(V1z_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;
    }

    if (PyArray_DIMS(V2x_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;
    }

    if (PyArray_DIMS(V2y_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;
    }

    if (PyArray_DIMS(V2z_np_array)[0]!=m){
        PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Dimensions mismatch.\n");
		return NULL;
    }


    mhn=PyArray_DIMS(M_np_array)[0];
    mh=(int)mhn;
	if (mh>m){
		PyErr_SetString(PyExc_ValueError,"nonmaxsup_stub_single.c: Mask dimensions mismatch.\n");
		printf("ERROR: nonmaxsup: Mask dimensions mismatch.\n");
		return NULL;
	}




	//Saving data in C
	double* I =(double*)PyArray_DATA(I_np_array);
	double* V1x=(double*)PyArray_DATA(V1x_np_array);
	double* V1y=(double*)PyArray_DATA(V1y_np_array);
	double* V1z=(double*)PyArray_DATA(V1z_np_array);
	double* V2x=(double*)PyArray_DATA(V2x_np_array);
	double* V2y=(double*)PyArray_DATA(V2y_np_array);
	double* V2z=(double*)PyArray_DATA(V2z_np_array);
	long long int* M=(long long int*)PyArray_DATA(M_np_array);
	unsigned int* dim=(unsigned int*)PyArray_DATA(dim_np_array);

	//Free memory
	Py_XDECREF(I_np_array); Py_XDECREF(V1x_np_array); Py_XDECREF(V1y_np_array); Py_XDECREF(V1z_np_array);
	Py_XDECREF(V2x_np_array); Py_XDECREF(V2y_np_array); Py_XDECREF(V2z_np_array); Py_XDECREF(M_np_array);
	Py_XDECREF(dim_np_array);

	// Get computer information to set the number of thread and the size of buffers
	num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (num_threads<1) {
		PyErr_SetString(PyExc_RuntimeError,"nonmaxsup_stub_single.cpp: No active cpu detected.\n");
		printf("ERROR: nonmaxsup: No active cpu detected.\n");
		return NULL;
	}

	//num_threads=1;
	dat64 = get_cache_size();
	if (dat64<1) {
		PyErr_SetString(PyExc_RuntimeError,"nonmaxsup_stub_single.cpp: No Cache detected.\n");
		printf("ERROR: nonmaxsup: No Cache detected.\n");
		return NULL;
	}
	task_size = ceil( (CACHE_USED*dat64) / 18 );

	
	
	ld = mh;

	// Create the array for holding the output result

	F= (unsigned int*)malloc(m*sizeof(unsigned int));

    Tomos tomo;
    tomo.I=I;
    tomo.V1x=V1x;
    tomo.V1y=V1y;
    tomo.V1z=V1z;
    tomo.V2x=V2x;
    tomo.V2y=V2y;
    tomo.V2z=V2z;
    tomo.M=M;
    tomo.dim=dim;

    tomo.F=F;

	// Assign pointers to data
	
	
	// Set pointer for initial splitting
	nta = (float)m / task_size;
	nta = ceil( nta );
	nth = num_threads;
	if (nta<nth) {
		nth = nta;
	}

	// Throw the workers
	lq = m;
	sq = 0; // Task queue initialization
	if (pthread_mutex_init( &mutex, NULL ) ){
		PyErr_SetString(PyExc_RuntimeError,"nonmaxsup_stub.c: Error creating the mutex.\n");
		printf("ERROR: nonmaxsup: Unable to create the mutex.\n");
		return NULL;
	}

	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (i=0; i<nth; i++) {
		// Update process queue pointers
		if (pthread_create(&threads[i],NULL,&look_neigbourhood,&tomo)) {
			PyErr_SetString(PyExc_RuntimeError,"nonmaxsup_stub.c: Error creating a thread.\n");
			printf("ERROR: nonmaxsup: Unable to create a thread.\n");
			return NULL;
		}
	}

	
	// Wait for all workers
	for (i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			PyErr_SetString(PyExc_RuntimeError,"nonmaxsup_stub.c: Error waiting the thread termination.\n");
			printf("ERROR: nonmaxsup: Fail waiting the thread termination.\n");
			return NULL;
		}
	}

	//Creating numpy matrix from C
	PyObject* F_array = PyArray_SimpleNewFromData(1, &mn, NPY_UINT32, tomo.F);
	if (F_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix F.\n");
        printf("ERROR: nonmaxsup: Fail to create NumPy Matrix F.\n");
        return NULL;
    }
	PyArray_ENABLEFLAGS((PyArrayObject*)F_array, NPY_ARRAY_OWNDATA);

	return F_array;

}
typedef struct{
	    //Inputs
	    double* Ixx;
	    double* Iyy;
	    double* Izz;
	    double* Ixy;
	    double* Ixz;
	    double* Iyz;

        //outputs
        double* L1;
        double* L2;
        double* L3;
        double* V1x;
        double* V1y;
        double* V1z;
        double* V2x;
        double* V2y;
        double* V2z;
        double* V3x;
        double* V3y;
        double* V3z;
        } Images;

static PyObject * supression_desyevv(PyObject *self, PyObject *args)
{
    //Inputs
    PyObject* Ixx_array;//double
    PyObject* Iyy_array;//double
    PyObject* Izz_array;//double
    PyObject* Ixy_array;//double
    PyObject* Ixz_array;//double
    PyObject* Iyz_array;//double
    //double *Ixx, *Iyy, *Izz, *Ixy, *Ixz, *Iyz;

    //Auxiliar variables
	int nta, nth, num_threads;
	//int i, type;
	int m;
	npy_intp mh;
	//size_t len, len64;
	long long int dat64;
	pthread_t* threads;
	//int sq, lq, ld, task_size;
	//pthread_mutex_t mutex;

	//Structura con datos de input e input
    double *L1, *L2, *L3;
    double *V1x, *V1y, *V1z, *V2x, *V2y, *V2z, *V3x, *V3y, *V3z;


    if (!PyArg_ParseTuple(args, "OOOOOO", &Ixx_array, &Iyy_array, &Izz_array, &Ixy_array, &Ixz_array, &Iyz_array)){
        printf("ERROR: desyevv: Unable to load inputs.\n");
        PyErr_SetString(PyExc_TypeError, "Unable to load inputs.\n");

        return NULL;
    }

    //Transform to NumPy matrix
	PyArrayObject* Ixx_np_array = (PyArrayObject*)PyArray_FROM_OTF(Ixx_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Ixx_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Ixx in a NumPy matrix.\n");
        printf("ERROR: desyevv: Unable transforming Ixx in a NumPy matrix.\n");
        return NULL;
    }

    PyArrayObject* Iyy_np_array = (PyArrayObject*)PyArray_FROM_OTF(Iyy_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Ixx_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Iyy in a NumPy matrix.\n");
        printf("ERROR: desyevv Unable transforming Iyy in a NumPy matrix.\n");
        return NULL;
    }

    PyArrayObject* Izz_np_array = (PyArrayObject*)PyArray_FROM_OTF(Izz_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Izz_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Izz in a NumPy matrix.\n");
        printf("ERROR: desyevv: Unable transforming Izz in a NumPy matrix.\n");
        return NULL;
    }

    PyArrayObject* Ixy_np_array = (PyArrayObject*)PyArray_FROM_OTF(Ixy_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Ixy_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Ixy in a NumPy matrix.\n");
        printf("ERROR: desyevv: Unable transforming Ixy in a NumPy matrix.\n");
        return NULL;
    }

    PyArrayObject* Ixz_np_array = (PyArrayObject*)PyArray_FROM_OTF(Ixz_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Ixz_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Ixz in a NumPy matrix.\n");
        printf("ERROR: desyevv: Unable transforming Ixz in a NumPy matrix.\n");
        return NULL;
    }

    PyArrayObject* Iyz_np_array = (PyArrayObject*)PyArray_FROM_OTF(Iyz_array, NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
	if (Iyz_np_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Error transforming Iyz in a NumPy matrix.\n");
        printf("ERROR: desyevv: Unable transforming Iyz in a NumPy matrix.\n");
        return NULL;
    }



    mh=PyArray_DIMS(Ixx_np_array)[0];
    m=(int)mh;


    if (PyArray_DIMS(Iyy_np_array)[0]!=m){
	    PyErr_SetString(PyExc_ValueError,"desyevvmex.c: Dimensions mismatch.\n");
		printf("ERROR: desyevv: Dimensions mismatch.\n");
		return NULL;
	}

	if (PyArray_DIMS(Izz_np_array)[0]!=m){
	    PyErr_SetString(PyExc_ValueError,"desyevvmex.c: Dimensions mismatch.\n");
		printf("ERROR: desyevv: Dimensions mismatch.\n");
		return NULL;
	}

	if (PyArray_DIMS(Ixy_np_array)[0]!=m){
	    PyErr_SetString(PyExc_ValueError,"desyevvmex.c: Dimensions mismatch.\n");
		printf("ERROR: desyevv: Dimensions mismatch.\n");
		return NULL;
	}

	if (PyArray_DIMS(Ixz_np_array)[0]!=m){
	    PyErr_SetString(PyExc_ValueError,"desyevvmex.c: Dimensions mismatch.\n");
		printf("ERROR: desyevv: Dimensions mismatch.\n");
		return NULL;
	}

	if (PyArray_DIMS(Iyz_np_array)[0]!=m){
	    PyErr_SetString(PyExc_ValueError,"desyevvmex.c: Dimensions mismatch.\n");
		printf("ERROR: desyevv: Dimensions mismatch.\n");
		return NULL;
	}



	ld=mh;

    double* Ixx =(double*)PyArray_DATA(Ixx_np_array);
    double* Iyy =(double*)PyArray_DATA(Iyy_np_array);
    double* Izz =(double*)PyArray_DATA(Izz_np_array);
    double* Ixy =(double*)PyArray_DATA(Ixy_np_array);
    double* Ixz =(double*)PyArray_DATA(Ixz_np_array);
    double* Iyz =(double*)PyArray_DATA(Iyz_np_array);





    //Free memory
	Py_XDECREF(Ixx_np_array); Py_XDECREF(Iyy_np_array); Py_XDECREF(Izz_np_array);
	Py_XDECREF(Ixy_np_array); Py_XDECREF(Ixz_np_array); Py_XDECREF(Iyz_np_array);



    // Get computer information to set the number of thread and the size of buffers
	num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (num_threads<1) {
		PyErr_SetString(PyExc_RuntimeError,"desyevvmex.c: No active cpu detected.\n");
		printf("ERROR: line desyevv: No active cpu detected.\n");
		return NULL;
	}

    //num_threads=1;
	dat64 = get_cache_size();
	if (dat64<1) {
		PyErr_SetString(PyExc_RuntimeError,"desyevvmex.c: No Cache L2 detected.\n");
		printf("ERROR: line desyevv: No Cache L2 detected.\n");
		return NULL;
	}
	task_size = ceil( (CACHE_USED*dat64) / 18 );

	//Create output c matrix

	L1=(double*)malloc(m*sizeof(double));
	L2=(double*)malloc(m*sizeof(double));
	L3=(double*)malloc(m*sizeof(double));
	V1x=(double*)malloc(m*sizeof(double));
	V1y=(double*)malloc(m*sizeof(double));
	V1z=(double*)malloc(m*sizeof(double));
	V2x=(double*)malloc(m*sizeof(double));
	V2y=(double*)malloc(m*sizeof(double));
	V2z=(double*)malloc(m*sizeof(double));
	V3x=(double*)malloc(m*sizeof(double));
	V3y=(double*)malloc(m*sizeof(double));
	V3z=(double*)malloc(m*sizeof(double));

    Images image;
    image.Ixx=Ixx;
    image.Iyy=Iyy;
    image.Izz=Izz;
    image.Ixy=Ixy;
    image.Ixz=Ixz;
    image.Iyz=Iyz;
    image.L1=L1;
    image.L2=L2;
    image.L3=L3;
    image.V1x=V1x;
    image.V1y=V1y;
    image.V1z=V1z;
    image.V2x=V2x;
    image.V2y=V2y;
    image.V2z=V2z;
    image.V3x=V3x;
    image.V3y=V3y;
    image.V3z=V3z;
    //double* images[18] = {Ixx,Iyy,Izz,Ixy,Ixz,Iyz,L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z};

	// Set pointer for initial splitting
	nta = (float)m / task_size;
	nta = ceil( nta );
	nth = num_threads;
	if (nta<nth) {
		nth = nta;
	}

	// Throw the workers
	lq = m;
	sq = 0; // Task queue initialization
	if (pthread_mutex_init( &mutex, NULL ) ){
		PyErr_SetString(PyExc_RuntimeError,"desyevvmex.cpp: Error creating the mutex.\n");
		printf("ERROR: desyevv: Unable to create the mutex.\n");
		return NULL;
	}

	threads = (pthread_t*)malloc( nth*sizeof(pthread_t) );
	for (int i=0; i<nth; i++) {
		// Update process queue pointers

		if (pthread_create(&threads[i],NULL,&desyevv3stub,&image)) {
			PyErr_SetString(PyExc_RuntimeError,"desyevvmex.cpp: Error creating a thread.\n");
			printf("ERROR: desyevv: Unable to create a thread.\n");
			return NULL;
		}
	}

	// Wait for all workers
	for (int i=0; i<nth; i++) {
		if (pthread_join(threads[i],NULL)) {
			PyErr_SetString(PyExc_RuntimeError,"desyevvmex.cpp: Error waiting the thread termination.\n");
			printf("ERROR: desyevv: Failure waiting the thread termination.\n");
			return NULL;
		}
	}



	//Desempaqueto la estructura


	//Save outputs as a mumpy array
	PyObject* L1_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.L1);


	if (L1_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix L1.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix L1.\n");
        return NULL;
    }



	PyArray_ENABLEFLAGS((PyArrayObject*)L1_array, NPY_ARRAY_OWNDATA);


	PyObject* L2_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.L2);
	if (L2_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix L2.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix L2.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)L2_array, NPY_ARRAY_OWNDATA);

	PyObject* L3_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.L3);
	if (L3_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix L3.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix L3.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)L3_array, NPY_ARRAY_OWNDATA);


	PyObject* V1x_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V1x);
	if (V1x_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V1x.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V1x.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V1x_array, NPY_ARRAY_OWNDATA);


	PyObject* V1y_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V1y);
	if (V1y_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V1y.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V1y.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V1y_array, NPY_ARRAY_OWNDATA);

	PyObject* V1z_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V1z);
	if (V1z_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V1z.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V1z.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V1z_array, NPY_ARRAY_OWNDATA);


	PyObject* V2x_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V2x);
	if (V2x_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V2x.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V2x.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V2x_array, NPY_ARRAY_OWNDATA);


	PyObject* V2y_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V2y);
	if (V2y_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V2y.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V2y.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V2y_array, NPY_ARRAY_OWNDATA);


	PyObject* V2z_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V2z);
	if (V2z_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V2z.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V2z.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V2z_array, NPY_ARRAY_OWNDATA);


	PyObject* V3x_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V3x);
	if (V3x_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V3x.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V3x.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V3x_array, NPY_ARRAY_OWNDATA);


	PyObject* V3y_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V3y);
	if (V3y_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V3y.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V3i.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V3y_array, NPY_ARRAY_OWNDATA);


	PyObject* V3z_array = PyArray_SimpleNewFromData(1, &mh, NPY_DOUBLE, image.V3z);
	if (V3z_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Fail to create NumPy Matrix V3z.\n");
        printf("ERROR: desyevv: Fail to create NumPy Matrix V3z.\n");
        return NULL;
    }

	PyArray_ENABLEFLAGS((PyArrayObject*)V3z_array, NPY_ARRAY_OWNDATA);


	PyObject *result = PyTuple_Pack(12, L1_array, L2_array, L3_array, V1x_array, V1y_array, V1z_array,
	    V2x_array, V2y_array, V2z_array, V3x_array, V3y_array, V3z_array);

	return result;
}

// Thread for calling the function for doing the calculations
static void* desyevv3stub( void* ptr ){


    typedef struct{
	    //Inputs
	    double* Ixx;
	    double* Iyy;
	    double* Izz;
	    double* Ixy;
	    double* Ixz;
	    double* Iyz;

        //outputs
        double* L1;
        double* L2;
        double* L3;
        double* V1x;
        double* V1y;
        double* V1z;
        double* V2x;
        double* V2y;
        double* V2z;
        double* V3x;
        double* V3y;
        double* V3z;
        } Images;

	int start, end;
	double A[3][3];
	double Q[3][3];
	double w[3];
	bool lock = true;


	//Cargo el pointer
	Images* image = (Images*)ptr;
    double* Ixx = image->Ixx;
    double* Iyy = image->Iyy;
    double* Izz = image->Izz;
    double* Ixy = image->Ixy;
    double* Ixz = image->Ixz;
    double* Iyz = image->Iyz;



	do{
		// Update pointers
		pthread_mutex_lock( &mutex );
		start = sq;
		sq = start + task_size;
		if (sq>=ld) {
			sq = ld;
			lock = false;
		}
		end = sq;
		pthread_mutex_unlock( &mutex );
        //printf("Mutex locked\n");
        //printf("start:%d\n",start);

        //printf("end:%d\n",end);

		// Precesing task
		for (int i=start; i<end; i++) {
			// Fill stub data
			A[0][0] = Ixx[i];
			//printf("Ixx: %f\n", Ixx);
			//printf("A[0][0]: %f\n",A[0][0]);
			A[0][1] = Ixy[i];
			//printf("A[0][1]: %f\n",A[0][1]);
			A[0][2] = Ixz[i];
			//printf("A[0][2]: %f\n",A[0][2]);
			A[1][0] = Ixy[i];
			//printf("A[1][0]: %f\n",A[1][0]);
			A[1][1] = Iyy[i];
			//printf("A[1][1]: %f\n",A[1][1]);
			A[1][2] = Iyz[i];
			//printf("A[1][2]: %f\n",A[1][2]);
			A[2][0] = Ixz[i];
			//printf("A[2][0]: %f\n",A[2][0]);
			A[2][1] = Iyz[i];
			//printf("A[2][1]: %f\n",A[2][1]);
			A[2][2] = Izz[i];
			//printf("A[2][2]: %f\n",A[2][2]);
			// Eigemproblem computation
			//printf("Empieza funcion\n");

			dsyevv3( A, Q, w );
			//printf("%d\n",dsyevv3( A, Q, w ));
			//printf("termina funcion\n");

			// Fill output arrays with the results
			image->V1x[i] = Q[0][0];
			//printf("Q[0][0]: %f\n",Q[0][0]);
			image->V1y[i] = Q[1][0];
			//printf("Q[1][0]: %f\n",Q[1][0]);
			image->V1z[i] = Q[2][0];
			//printf("Q[2][0]: %f\n",Q[2][0]);
			image->V2x[i] = Q[0][1];
			//printf("Q[0][1]: %f\n",Q[0][1]);
			image->V2y[i] = Q[1][1];
			//printf("Q[1][1]: %f\n",Q[1][1]);
			image->V2z[i] = Q[2][1];
			//printf("Q[2][1]: %f\n",Q[2][1]);
			image->V3x[i] = Q[0][2];
			//printf("Q[0][2]: %f\n",Q[0][2]);
			image->V3y[i] = Q[1][2];
			//printf("Q[1][2]: %f\n",Q[1][2]);
			image->V3z[i] = Q[2][2];
			//printf("Q[2][2]: %f\n",Q[2][2]);
			image->L1[i] = w[0];
			//printf("w[0]: %f\n",w[0]);
			image->L2[i] = w[1];
			//printf("w[1]: %f\n",w[1]);
			image->L3[i] = w[2];
			//printf("w[2]: %f\n",w[2]);

            //printf("index %d \n",i);
		}
	}while(lock);

}

// ----------------------------------------------------------------------------
static int dsyevv3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Qo: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
	double norm;          // Squared norm or inverse norm of current eigenvector
	double n0, n1;        // Norm of first and second columns of A
	double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
	double thresh;        // Small number used as threshold for floating point comparisons
	double error;         // Estimated maximum roundoff error in some steps
	double wmax;          // The eigenvalue of maximum modulus
	double f, t;          // Intermediate storage
	int i, j;             // Loop counters
	double hold;
#endif

	// Calculate eigenvalues
	dsyevc3(A, w);

	// Ordering eigenvalues (buble sort)
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
	}
	if ( fabs(w[2]) > fabs(w[1]) ) {
		hold = w[2];
		w[2] = w[1];
		w[1] = hold;
	}
	if ( fabs(w[1]) > fabs(w[0]) ) {
		hold = w[1];
		w[1] = w[0];
		w[0] = hold;
	}

#ifndef EVALS_ONLY
	wmax = fabs(w[0]);
	if ((t=fabs(w[1])) > wmax)
		wmax = t;
	if ((t=fabs(w[2])) > wmax)
		wmax = t;
	thresh = SQR(8.0 * DBL_EPSILON * wmax);

	// Prepare calculation of eigenvectors
	n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
	n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = SQR(A[0][1]);

	// Calculate first eigenvector by the formula
	//   v[0] = (A - w[0]).e1 x (A - w[0]).e2
	A[0][0] -= w[0];
	A[1][1] -= w[0];
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
	norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
	n0      = n0tmp + SQR(A[0][0]);
	n1      = n1tmp + SQR(A[1][1]);
	error   = n0 * n1;

	if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
	{
		Q[0][0] = 1.0;
		Q[1][0] = 0.0;
		Q[2][0] = 0.0;
	}
	else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
	{
		Q[0][0] = 0.0;
		Q[1][0] = 1.0;
		Q[2][0] = 0.0;
	}
	else if (norm < SQR(64.0 * DBL_EPSILON) * error)
	{                         // If angle between A[0] and A[1] is too small, don't use
		t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
		f = -A[0][0] / A[0][1];
		if (SQR(A[1][1]) > t)
		{
			t = SQR(A[1][1]);
			f = -A[0][1] / A[1][1];
		}
		if (SQR(A[1][2]) > t)
			f = -A[0][2] / A[1][2];
		norm    = 1.0/sqrt(1 + SQR(f));
		Q[0][0] = norm;
		Q[1][0] = f * norm;
		Q[2][0] = 0.0;
	}
	else                      // This is the standard branch
	{
		norm = sqrt(1.0 / norm);
		for (j=0; j < 3; j++)
			Q[j][0] = Q[j][0] * norm;
	}


	// Prepare calculation of second eigenvector
	t = w[0] - w[1];
	if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
	{
		// For non-degenerate eigenvalue, calculate second eigenvector by the formula
		//   v[1] = (A - w[1]).e1 x (A - w[1]).e2
		A[0][0] += t;
		A[1][1] += t;
		Q[0][1]  = Q[0][1] + A[0][2]*w[1];
		Q[1][1]  = Q[1][1] + A[1][2]*w[1];
		Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
		norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
		n0       = n0tmp + SQR(A[0][0]);
		n1       = n1tmp + SQR(A[1][1]);
		error    = n0 * n1;

		if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
		{
			Q[0][1] = 1.0;
			Q[1][1] = 0.0;
			Q[2][1] = 0.0;
		}
		else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
		{
			Q[0][1] = 0.0;
			Q[1][1] = 1.0;
			Q[2][1] = 0.0;
		}
		else if (norm < SQR(64.0 * DBL_EPSILON) * error)
		{                       // If angle between A[0] and A[1] is too small, don't use
			t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
			f = -A[0][0] / A[0][1];
			if (SQR(A[1][1]) > t)
			{
				t = SQR(A[1][1]);
				f = -A[0][1] / A[1][1];
			}
			if (SQR(A[1][2]) > t)
				f = -A[0][2] / A[1][2];
			norm    = 1.0/sqrt(1 + SQR(f));
			Q[0][1] = norm;
			Q[1][1] = f * norm;
			Q[2][1] = 0.0;
		}
		else
		{
			norm = sqrt(1.0 / norm);
			for (j=0; j < 3; j++)
				Q[j][1] = Q[j][1] * norm;
		}
	}
	else
	{
		// For degenerate eigenvalue, calculate second eigenvector according to
		//   v[1] = v[0] x (A - w[1]).e[i]
		//
		// This would really get to complicated if we could not assume all of A to
		// contain meaningful values.
		A[1][0]  = A[0][1];
		A[2][0]  = A[0][2];
		A[2][1]  = A[1][2];
		A[0][0] += w[0];
		A[1][1] += w[0];
		for (i=0; i < 3; i++)
		{
			A[i][i] -= w[1];
			n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
			if (n0 > thresh)
			{
				Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
				Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
				Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
				norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
				if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
				{                                         // the two vectors was not too small
					norm = sqrt(1.0 / norm);
					for (j=0; j < 3; j++)
						Q[j][1] = Q[j][1] * norm;
					break;
				}
			}
		}

		if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
		{
			for (j=0; j < 3; j++)
				if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
				{                                                     // ... and swap it with the next one
					norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
					Q[j][1]       = Q[(j+1)%3][0] * norm;
					Q[(j+1)%3][1] = -Q[j][0] * norm;
					Q[(j+2)%3][1] = 0.0;
					break;
				}
		}
	}


	// Calculate third eigenvector according to
	//   v[2] = v[0] x v[1]
	Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
	Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
	Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
#endif

	return 0;
}

// ----------------------------------------------------------------------------
static int dsyevc3(double A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
	double m, c1, c0;

	// Determine coefficients of characteristic poynomial. We write
	//       | a   d   f  |
	//  A =  | d*  b   e  |
	//       | f*  e*  c  |
	double de = A[0][1] * A[1][2];                                    // d * e
	double dd = SQR(A[0][1]);                                         // d^2
	double ee = SQR(A[1][2]);                                         // e^2
	double ff = SQR(A[0][2]);                                         // f^2
	m  = A[0][0] + A[1][1] + A[2][2];
	c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
	- (dd + ee + ff);
	c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
	- 2.0 * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

	double p, sqrt_p, q, c, s, phi;
	p = SQR(m) - 3.0*c1;
	q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
	sqrt_p = sqrt(fabs(p));

	phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
	phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);

	c = sqrt_p*cos(phi);
	s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

	w[1]  = (1.0/3.0)*(m - c);
	w[2]  = w[1] + s;
	w[0]  = w[1] + c;
	w[1] -= s;

	return 0;
}

// Thread for measuring the intermediate neighbour value
static void* look_neigbourhood( void* ptr ){


    typedef struct{
//inputs
    double* I;
	double* V1x;
	double* V1y;
	double* V1z;
	double* V2x;
	double* V2y;
	double* V2z;
	long long int* M;
	unsigned int* dim;

    //Output
	unsigned int* F ;
    }Tomos;
	
	int i, j, k;
	unsigned int mx, my ;
	//unsigned int *dim;
	int sz, start, end;
	double lv1, hold1, k1x, k1y, k1z;
	double lv2, hold2, k2x, k2y, k2z;
	double* A1[8];
	double* B1[8];
	double* A2[8];
	double* B2[8];
	double* V1a;
	double* V1b;
	double* V2a;
	double* V2b;
	double** K1;
	double** K2;
	unsigned char lock = 0x01;
	Tomos* tomo = (Tomos*)ptr;
	double* I = tomo->I;
	double* V1x = tomo->V1x;
	double* V1y = tomo->V1y;
	double* V1z = tomo->V1z;
	double* V2x = tomo->V2x;
	double* V2y = tomo->V2y;
	double* V2z = tomo->V2z;
	long long int* M = tomo->M;
	unsigned int* dim = tomo->dim;
	mx = dim[0];
	my = dim[1];

	// Buffers initialization
	sz = task_size * sizeof(double);
	for (i=0; i<8; i++) {
		A1[i] = (double*)malloc( sz );
		B1[i] = (double*)malloc( sz );
		A2[i] = (double*)malloc( sz );
		B2[i] = (double*)malloc( sz );
	}	
	V1a = (double*)malloc( sz );
	V1b = (double*)malloc( sz );
	K1 = (double**)malloc( 3*sizeof(double*) );
	V2a = (double*)malloc( sz );
	V2b = (double*)malloc( sz );
	K2 = (double**)malloc( 3*sizeof(double*) );
	for (i=0; i<3; i++) {
		K1[i] = (double*)malloc( sz );
		K2[i] = (double*)malloc( sz );
	}
	
	// Task loop
	do{
		// Update pointers
		pthread_mutex_lock( &mutex );
		start = sq;
		sq = start + task_size;	
		if (sq>=ld) {
			sq = ld;
			lock = 0x00;
		}
		end = sq;
		pthread_mutex_unlock( &mutex );		
		
		// Prepare data for every coordinate
		j = 0;
		for (k=start; k<end; k++) {

			i = M[k];
			//printf("k=%d, i=%d, j=%d\n",k,i,j);



			lv1 = I[i];
			//printf("I[%d] vale %f\n",i,I[i]);
			//printf("lv1 vale %f\n",lv1);

			K1[0][j] = fabs( V1x[i] * INTER_FACTOR );
			K1[1][j] = fabs( V1y[i] * INTER_FACTOR );
			K1[2][j] = fabs( V1z[i] * INTER_FACTOR );

			//printf("V1x[0] vale %f\n",V1x[0]);
	        //printf("V1y[0] vale %f\n",V1y[0]);
	        //printf("V1z[0] vale %f\n",V1z[0]);
	        //printf("K1[0][j] vale %f\n",K1[0][j]);
	        //printf("K1[1][j] vale %f\n",K1[1][j]);
	        //printf("K1[2][j] vale %f\n",K1[2][j]);

			A1[0][j] = lv1;
			B1[0][j] = lv1;
			if (V1x[i]>=0) {
				A1[1][j] = I[i+mx*my];
				B1[1][j] = I[i-mx*my];
				if ( (V1y[i]>=0) && (V1z[i]>=0) ) {
					A1[2][j] = I[i+mx];
					A1[3][j] = I[i+mx*(my+1)];
					A1[4][j] = I[i+1];
					A1[5][j] = I[i+mx*my+1];
					A1[6][j] = I[i+mx+1];
					A1[7][j] = I[i+mx*(my+1)+1];
					B1[2][j] = I[i-mx];
					B1[3][j] = I[i-mx*(my+1)];
					B1[4][j] = I[i-1];
					B1[5][j] = I[i-mx*my-1];
					B1[6][j] = I[i-mx-1];					
					B1[7][j] = I[i-mx*(my+1)-1];
				}else if ( (V1y[i]<0) && (V1z[i]>=0) ) {
					A1[2][j] = I[i-mx];
					A1[3][j] = I[i+mx*(my-1)];
					A1[4][j] = I[i+1];
					A1[5][j] = I[i+mx*my+1];
					A1[6][j] = I[i-mx+1];
					A1[7][j] = I[i+mx*(my-1)+1];
					B1[2][j] = I[i+mx];
					B1[3][j] = I[i-mx*(my-1)];
					B1[4][j] = I[i-1];
					B1[5][j] = I[i-mx*my-1];
					B1[6][j] = I[i+mx-1];
					B1[7][j] = I[i-mx*(my-1)-1];
				}else if ( (V1y[i]>=0) && (V1z[i]<0) ) {
					A1[2][j] = I[i+mx];
					A1[3][j] = I[i+mx*(my+1)];
					A1[4][j] = I[i+1];
					A1[5][j] = I[i+mx*my+1];
					A1[6][j] = I[i+mx+1];
					A1[7][j] = I[i+mx*(my+1)+1];
					B1[2][j] = I[i-mx];
					B1[3][j] = I[i-mx*(my+1)];
					B1[4][j] = I[i-1];
					B1[5][j] = I[i-mx*my-1];
					B1[6][j] = I[i-mx-1];
					B1[7][j] = I[i-mx*(my+1)-1];
				}else {
					A1[2][j] = I[i-mx];
					A1[3][j] = I[i+mx*(my-1)];
					A1[4][j] = I[i+1];
					A1[5][j] = I[i+mx*my+1];
					A1[6][j] = I[i-mx+1];
					A1[7][j] = I[i+mx*(my-1)+1];
					B1[2][j] = I[i+mx];
					B1[3][j] = I[i-mx*(my-1)];
					B1[4][j] = I[i-1];
					B1[5][j] = I[i-mx*my-1];
					B1[6][j] = I[i+mx-1];
					B1[7][j] = I[i-mx*(my-1)-1];
				}
			}else {
				A1[1][j] = I[i-mx*my];
				B1[1][j] = I[i+mx*my];
				if ( (V1y[i]>=0) && (V1z[i]>=0) ) {
					A1[2][j] = I[i+mx];
					A1[3][j] = I[i-mx*(my-1)];
					A1[4][j] = I[i-1];
					A1[5][j] = I[i-mx*my-1];
					A1[6][j] = I[i+mx-1];
					A1[7][j] = I[i-mx*(my-1)-1];
					B1[2][j] = I[i-mx];
					B1[3][j] = I[i+mx*(my-1)];
					B1[4][j] = I[i+1];
					B1[5][j] = I[i+mx*my+1];
					B1[6][j] = I[i-mx+1];
					B1[7][j] = I[i+mx*(my-1)+1];
				}else if ( (V1y[i]<0) && (V1z[i]>=0) ) {
					A1[2][j] = I[i-mx];
					A1[3][j] = I[i-mx*(my+1)];
					A1[4][j] = I[i-1];
					A1[5][j] = I[i-mx*my-1];
					A1[6][j] = I[i-mx-1];
					A1[7][j] = I[i-mx*(my+1)-1];
					B1[2][j] = I[i+mx];
					B1[3][j] = I[i+mx*(my+1)];
					B1[4][j] = I[i+1];
					B1[5][j] = I[i+mx*my+1];
					B1[6][j] = I[i+mx+1];
					B1[7][j] = I[i+mx*(my+1)+1];
				}else if ( (V1y[i]>=0) && (V1z[i]<0) ) {
					A1[2][j] = I[i+mx];
					A1[3][j] = I[i-mx*(my-1)];
					A1[4][j] = I[i-1];
					A1[5][j] = I[i-mx*my-1];
					A1[6][j] = I[i+mx-1];
					A1[7][j] = I[i-mx*(my-1)-1];
					B1[2][j] = I[i-mx];
					B1[3][j] = I[i+mx*(my-1)];
					B1[4][j] = I[i+1];
					B1[5][j] = I[i+mx*my+1];
					B1[6][j] = I[i-mx+1];
					B1[7][j] = I[i+mx*(my-1)+1];
				}else {
					A1[2][j] = I[i-mx];
					A1[3][j] = I[i-mx*(my+1)];
					A1[4][j] = I[i-1];
					A1[5][j] = I[i-mx*my-1];
					A1[6][j] = I[i-mx-1];					
					A1[7][j] = I[i-mx*(my+1)-1];
					B1[2][j] = I[i+mx];
					B1[3][j] = I[i+mx*(my+1)];
					B1[4][j] = I[i+1];
					B1[5][j] = I[i+mx*my+1];
					B1[6][j] = I[i+mx+1];
					B1[7][j] = I[i+mx*(my+1)+1];
				}
			}				

		j++;}
		
		// Trilinear interpolation
		for (j=0; j<(end-start); j++) {
			k1x = K1[0][j];
			k1y = K1[1][j];
			k1z = K1[2][j];
			hold1 = A1[0][j]*(1-k1x)*(1-k1y)*(1-k1z);
			hold1 = hold1 + A1[4][j]*k1x*(1-k1y)*(1-k1z);
			hold1 = hold1 + A1[2][j]*(1-k1x)*k1y*(1-k1z);
			hold1 = hold1 + A1[1][j]*(1-k1x)*(1-k1y)*k1z;
			hold1 = hold1 + A1[5][j]*k1x*(1-k1y)*k1z;
			hold1 = hold1 + A1[3][j]*(1-k1x)*k1y*k1z;
			hold1 = hold1 + A1[6][j]*k1x*k1y*(1-k1z);
			V1a[j] = hold1 + A1[7][j]*k1x*k1y*k1z;
			//printf("V1a[j]=%f\n",V1a[j]);
		}
		for (j=0; j<(end-start); j++) {
			k1x = K1[0][j];
			k1y = K1[1][j];
			k1z = K1[2][j];
			hold1 = B1[0][j]*(1-k1x)*(1-k1y)*(1-k1z);
			hold1 = hold1 + B1[4][j]*k1x*(1-k1y)*(1-k1z);		
			hold1 = hold1 + B1[2][j]*(1-k1x)*k1y*(1-k1z);		
			hold1 = hold1 + B1[1][j]*(1-k1x)*(1-k1y)*k1z;			
			hold1 = hold1 + B1[5][j]*k1x*(1-k1y)*k1z;		
			hold1 = hold1 + B1[3][j]*(1-k1x)*k1y*k1z;		
			hold1 = hold1 + B1[6][j]*k1x*k1y*(1-k1z);		
			V1b[j] = hold1 + B1[7][j]*k1x*k1y*k1z;

			//printf("V1b[j]=%f\n",V1b[j]);
		}
		
		
			// Prepare data for every coordinate
		j = 0;
		for (k=start; k<end; k++) {
			i = M[k];

			lv2 = I[i];
			K2[0][j] = fabs( V2x[i] * INTER_FACTOR );
			K2[1][j] = fabs( V2y[i] * INTER_FACTOR );
			K2[2][j] = fabs( V2z[i] * INTER_FACTOR );
			A1[0][j] = lv2;
			B1[0][j] = lv2;
			if (V2x[i]>=0) {
				A2[1][j] = I[i+mx*my];
				B2[1][j] = I[i-mx*my];
				if ( (V2y[i]>=0) && (V2z[i]>=0) ) {
					A2[2][j] = I[i+mx];
					A2[3][j] = I[i+mx*(my+1)];
					A2[4][j] = I[i+1];
					A2[5][j] = I[i+mx*my+1];
					A2[6][j] = I[i+mx+1];
					A2[7][j] = I[i+mx*(my+1)+1];
					B2[2][j] = I[i-mx];
					B2[3][j] = I[i-mx*(my+1)];
					B2[4][j] = I[i-1];
					B2[5][j] = I[i-mx*my-1];
					B2[6][j] = I[i-mx-1];					
					B2[7][j] = I[i-mx*(my+1)-1];
				}else if ( (V2y[i]<0) && (V2z[i]>=0) ) {
					A2[2][j] = I[i-mx];
					A2[3][j] = I[i+mx*(my-1)];
					A2[4][j] = I[i+1];
					A2[5][j] = I[i+mx*my+1];
					A2[6][j] = I[i-mx+1];
					A2[7][j] = I[i+mx*(my-1)+1];
					B2[2][j] = I[i+mx];
					B2[3][j] = I[i-mx*(my-1)];
					B2[4][j] = I[i-1];
					B2[5][j] = I[i-mx*my-1];
					B2[6][j] = I[i+mx-1];
					B2[7][j] = I[i-mx*(my-1)-1];
				}else if ( (V2y[i]>=0) && (V2z[i]<0) ) {
					A2[2][j] = I[i+mx];
					A2[3][j] = I[i+mx*(my+1)];
					A2[4][j] = I[i+1];
					A2[5][j] = I[i+mx*my+1];
					A2[6][j] = I[i+mx+1];
					A2[7][j] = I[i+mx*(my+1)+1];
					B2[2][j] = I[i-mx];
					B2[3][j] = I[i-mx*(my+1)];
					B2[4][j] = I[i-1];
					B2[5][j] = I[i-mx*my-1];
					B2[6][j] = I[i-mx-1];
					B2[7][j] = I[i-mx*(my+1)-1];
				}else {
					A2[2][j] = I[i-mx];
					A2[3][j] = I[i+mx*(my-1)];
					A2[4][j] = I[i+1];
					A2[5][j] = I[i+mx*my+1];
					A2[6][j] = I[i-mx+1];
					A2[7][j] = I[i+mx*(my-1)+1];
					B2[2][j] = I[i+mx];
					B2[3][j] = I[i-mx*(my-1)];
					B2[4][j] = I[i-1];
					B2[5][j] = I[i-mx*my-1];
					B2[6][j] = I[i+mx-1];
					B2[7][j] = I[i-mx*(my-1)-1];
				}
			}else {
				A2[1][j] = I[i-mx*my];
				B2[1][j] = I[i+mx*my];
				if ( (V2y[i]>=0) && (V2z[i]>=0) ) {
					A2[2][j] = I[i+mx];
					A2[3][j] = I[i-mx*(my-1)];
					A2[4][j] = I[i-1];
					A2[5][j] = I[i-mx*my-1];
					A2[6][j] = I[i+mx-1];
					A2[7][j] = I[i-mx*(my-1)-1];
					B2[2][j] = I[i-mx];
					B2[3][j] = I[i+mx*(my-1)];
					B2[4][j] = I[i+1];
					B2[5][j] = I[i+mx*my+1];
					B2[6][j] = I[i-mx+1];
					B2[7][j] = I[i+mx*(my-1)+1];
				}else if ( (V2y[i]<0) && (V2z[i]>=0) ) {
					A2[2][j] = I[i-mx];
					A2[3][j] = I[i-mx*(my+1)];
					A2[4][j] = I[i-1];
					A2[5][j] = I[i-mx*my-1];
					A2[6][j] = I[i-mx-1];
					A2[7][j] = I[i-mx*(my+1)-1];
					B2[2][j] = I[i+mx];
					B2[3][j] = I[i+mx*(my+1)];
					B2[4][j] = I[i+1];
					B2[5][j] = I[i+mx*my+1];
					B2[6][j] = I[i+mx+1];
					B2[7][j] = I[i+mx*(my+1)+1];
				}else if ( (V2y[i]>=0) && (V2z[i]<0) ) {
					A2[2][j] = I[i+mx];
					A2[3][j] = I[i-mx*(my-1)];
					A2[4][j] = I[i-1];
					A2[5][j] = I[i-mx*my-1];
					A2[6][j] = I[i+mx-1];
					A2[7][j] = I[i-mx*(my-1)-1];
					B2[2][j] = I[i-mx];
					B2[3][j] = I[i+mx*(my-1)];
					B2[4][j] = I[i+1];
					B2[5][j] = I[i+mx*my+1];
					B2[6][j] = I[i-mx+1];
					B2[7][j] = I[i+mx*(my-1)+1];
				}else {
					A2[2][j] = I[i-mx];
					A2[3][j] = I[i-mx*(my+1)];
					A2[4][j] = I[i-1];
					A2[5][j] = I[i-mx*my-1];
					A2[6][j] = I[i-mx-1];					
					A2[7][j] = I[i-mx*(my+1)-1];
					B2[2][j] = I[i+mx];
					B2[3][j] = I[i+mx*(my+1)];
					B2[4][j] = I[i+1];
					B2[5][j] = I[i+mx*my+1];
					B2[6][j] = I[i+mx+1];
					B2[7][j] = I[i+mx*(my+1)+1];
				}
			}				

		j++;}
		
		// Trilinear interpolation
		for (j=0; j<(end-start); j++) {
			k2x = K2[0][j];
			k2y = K2[1][j];
			k2z = K2[2][j];
			hold2 = A2[0][j]*(1-k2x)*(1-k2y)*(1-k2z);
			hold2 = hold2 + A2[4][j]*k2x*(1-k2y)*(1-k2z);
			hold2 = hold2 + A2[2][j]*(1-k2x)*k2y*(1-k2z);
			hold2 = hold2 + A2[1][j]*(1-k2x)*(1-k2y)*k2z;
			hold2 = hold2 + A2[5][j]*k1x*(1-k1y)*k1z;
			hold2 = hold2 + A2[3][j]*(1-k1x)*k1y*k1z;
			hold2 = hold2 + A2[6][j]*k2x*k2y*(1-k2z);
			V2a[j] = hold2 + A2[7][j]*k2x*k2y*k2z;			
		}
		for (j=0; j<(end-start); j++) {
			k2x = K2[0][j];
			k2y = K2[1][j];
			k2z = K2[2][j];
			hold2 = B2[0][j]*(1-k2x)*(1-k2y)*(1-k2z);
			hold2 = hold2 + B2[4][j]*k2x*(1-k2y)*(1-k2z);		
			hold2 = hold2 + B2[2][j]*(1-k2x)*k2y*(1-k2z);		
			hold2 = hold2 + B2[1][j]*(1-k2x)*(1-k2y)*k2z;			
			hold2 = hold2 + B2[5][j]*k2x*(1-k2y)*k2z;		
			hold2 = hold2 + B2[3][j]*(1-k2x)*k2y*k2z;		
			hold2 = hold2 + B2[6][j]*k2x*k2y*(1-k2z);		
			V2b[j] = hold2 + B2[7][j]*k2x*k2y*k2z;	
		}
		
		// Mark local maxima
		j = 0;
		for (k=start; k<end; k++) {
			i = M[k];

			lv1 = I[i];
			if ( (lv1>V1a[j]) && (lv1>V1b[j]) && (lv1>V2a[j]) && (lv1>V2b[j]) ) {

				tomo->F[i] = 1;

			}			

		j++;}
		
	}while(lock);
	
	pthread_exit(0);
}


// Get the cache size found in the first processor listed in /proc/cpuinfo in bytes 
static long long int get_cache_size() 
{
	FILE* fp; 
	char *match; 
	char buffer[BUFFER_SIZE];
	size_t bytes_read; 
	long int cache_size; 
	
	fp = fopen( "/proc/cpuinfo", "r" );
	bytes_read = fread(buffer, 1, BUFFER_SIZE, fp); 
	fclose (fp); 
	if( bytes_read <= 0 ) 
		return ERROR_FILE_READ;
	buffer[bytes_read] == '\0';
	match = strstr( buffer, "cache size" ); 
	if (match == NULL) 
		return ERROR_CACHE_NOT_FOUND;  
	sscanf( match, "cache size : %ld", &cache_size ); 
	
	return ((long long int)cache_size) * BYTE_PER_KBYTE; }
/*
static int get_M(double *mat[]) {
	return sizeof(mat)/sizeof(mat[0]);}
static int get_N(int *mat[]) {
	return sizeof(mat[0])/sizeof(mat[0][0]);}
*/

//Configure the module	
PyDoc_STRVAR(supression_doc, "Remove and return the rightmost element.");
//Method table
static PyMethodDef supressionMethods[]= {
    {"desyevv",supression_desyevv,METH_VARARGS,"Resolve eigenproblem mix method"},
	{"nonmaxsup", supression_nonmaxsup,METH_VARARGS,"Execute supression of non maximums"},
	{NULL,NULL,0,NULL}
};


//Module definition
static struct PyModuleDef supressionmodule = {
	PyModuleDef_HEAD_INIT,
	"supression",	// name of the module /
	supression_doc,	// module documentation, may be NULL 
	-1, //size of per-interpreter state of the module,
             //or -1 if the module keeps state in global variables.
    supressionMethods
};

//Init function
PyMODINIT_FUNC
PyInit_supression(void)
{

    assert(! PyErr_Occurred());
    import_array(); // Initialise Numpy
    if (PyErr_Occurred()) {
        return NULL;
    }
	return PyModule_Create(&supressionmodule);

}
