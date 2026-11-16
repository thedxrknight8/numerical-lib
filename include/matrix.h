#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} matrix;

int matrix_init(matrix* src, int rows, int cols);
int matrix_fill_random(matrix* src);
int matrix_fill_zeroes(matrix* src);
int matrix_destroy(matrix* src);
void print_matrix(const matrix* src);
int matrix_compare(const matrix* lhs, const matrix* rhs);

// OPS
int matrix_add_ij(matrix* dest, const matrix* a, const matrix* b);
int matrix_add_ji(matrix* dest, const matrix* a, const matrix* b);

int matmul_ijk(matrix* dest, const matrix *a, const matrix *b);
int matmul_ikj(matrix* dest, const matrix *a, const matrix *b);
int matmul_jik(matrix* dest, const matrix *a, const matrix *b);

int matmul_tiled(matrix* dest, const matrix *a, const matrix *b, int T);

int dot_product(const matrix* v1, const matrix* v2, float* result);
int hadamard(matrix* dest, const matrix* a, const matrix* b);

int matvec(matrix* dest, const matrix* a, const matrix* b);

int softmax(matrix* dest, const matrix* a, int axis);
int layernorm(matrix* dest, const matrix* a, const float* gamma, const size_t gamma_length, const float* beta, const size_t beta_length, const float eps);
int rmsnorm(matrix *dest, const matrix* a, const float* gamma, const size_t gamma_length, const float eps);
int silu(matrix* dest, matrix* a);

int transpose(matrix* dest, const matrix* src);




#endif