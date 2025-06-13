// This file is made by modifying "svm.h" in LIBSVM.

#ifndef _LIBLINEAR_KERNEL_H
#define _LIBLINEAR_KERNEL_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef float Qfloat;
typedef signed char schar;

#ifdef __cplusplus
extern "C" {
#endif

// kernel_type
// Note: LINEAR_KERNEL must be used only in the command line option.
// In other parts,
// - If LINEAR_KERNEL in the command line option, it should be replaced with LINEAR.
// - If LINEAR in the command line option, the "kernel_parameter" struct must be null,
//   and model::kernelized (in linear.h) must be zero.
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED, LINEAR_KERNEL };
extern const char *kernel_type_table[];

struct feature_node;
struct problem;

struct kernel_parameter
{
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
};

#ifdef __cplusplus
} /* extern "C" */
#endif

class Cache
{
public:
	Cache(int l,size_t size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);
private:
	int l;
	size_t size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, feature_node * const * x, const kernel_parameter& param);
	virtual ~Kernel();

	static double k_function(const feature_node *x, const feature_node *y,
				 const kernel_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const;

protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const feature_node **x;
	double *x_square;

	// kernel_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const feature_node *px, const feature_node *py);
	double kernel_linear(int i, int j) const;
	double kernel_poly(int i, int j) const;
	double kernel_rbf(int i, int j) const;
	double kernel_sigmoid(int i, int j) const;
	double kernel_precomputed(int i, int j) const;
};
//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{
public:
	SVC_Q(const problem& prob, const kernel_parameter& param, const schar *y_);
	Qfloat *get_Q(int i, int len) const;
	double *get_QD() const;
	void swap_index(int i, int j) const;
	~SVC_Q();
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class SVC_Q_NoCache: public Kernel
{
public:
	SVC_Q_NoCache(const problem& prob, const kernel_parameter& param, const schar *y_);
	Qfloat *get_Q(int i, int len) const;
	void get_Q(int i, int len, Qfloat *data) const;
	double *get_QD() const;
	void swap_index(int i, int j) const;
	~SVC_Q_NoCache();
private:
	schar *y;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const problem& prob, const kernel_parameter& param);
	Qfloat *get_Q(int i, int len) const;
	double *get_QD() const;
	void swap_index(int i, int j) const;
	~ONE_CLASS_Q();
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{
public:
	SVR_Q(const problem& prob, const kernel_parameter& param);
	void swap_index(int i, int j) const;
	Qfloat *get_Q(int i, int len) const;
	double *get_QD() const;
	~SVR_Q();
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

#endif /* _LIBLINEAR_KERNEL_H */
