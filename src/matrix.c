#include "matrix.h"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int matrix_init(matrix *src, int rows, int cols) {
    if(src == NULL || rows <= 0 || cols <= 0) {
        return 0;
    }

    float* data = (float*)malloc(sizeof(float) * rows * cols);
    if(data == NULL) {
        return 0;
    }

    src->rows = rows;
    src->cols = cols;
    src->data = data;
    return 1;
}

int matrix_fill_random(matrix *src) {
    if(src == NULL || src->data == NULL) {return 0;}
    for(size_t i = 0; i < src->rows; i++) {
        for(size_t j = 0; j < src->cols; j++) {
            src->data[i * src->cols + j] = (float)((double)rand() / (double) RAND_MAX);
        }
    }

    return 1;
}

int matrix_fill_zeroes(matrix *src) {
    if(src == NULL || src->data == NULL) {return 0;}
    for(size_t i = 0; i < src->rows; i++) {
        for(size_t j = 0; j < src->cols; j++) {
            src->data[i * src->cols + j] = 0.0f;
        }
    }

    return 1;
}

// ---- MATRIX ADD ----

int matrix_add_ij(matrix *dest, const matrix *a, const matrix *b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->cols || a->rows != b->rows) {return 0;}
    if(dest->rows != b->rows || dest->cols != b->cols) {return 0;}

    for(size_t i = 0; i < a->rows; i++) {
        for(size_t j = 0; j < a->cols; j++) {
            dest->data[i * dest->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
        }
    }

    return 1;
}

int matrix_add_ji(matrix *dest, const matrix *a, const matrix *b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->cols || a->rows != b->rows) {return 0;}
    if(dest->rows != b->rows || dest->cols != b->cols) {return 0;}

    for(size_t j = 0; j < a->cols; j++) {
        for(size_t i = 0; i < a->rows; i++) {
            dest->data[i * dest->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
        }
    }

    return 1;
}

int hadamard(matrix* dest, const matrix* a, const matrix* b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->cols || a->rows != b->rows) {return 0;}
    if(dest->rows != b->rows || dest->cols != b->cols) {return 0;}

    for(size_t i = 0; i < a->rows; i++) {
        for(size_t j = 0; j < a->cols; j++) {
            dest->data[i * dest->cols + j] = a->data[i * a->cols + j] * b->data[i * b->cols + j];
        }
    }

    return 1;
}

int matrix_destroy(matrix *src) {
    if(src == NULL || src->data == NULL) {
        return 0;
    }

    free(src->data);
    *src = (matrix){0};
    return 1;
}

void print_matrix(const matrix *src) {
    for(size_t i = 0; i < src->rows; i++) {
        for(size_t j = 0; j < src->cols; j++) {
            printf("%.6f ", src->data[i * src->cols + j]);
        }

        printf("\n");
    }
    printf("\n");
}

int dot_product(const matrix *v1, const matrix *v2, float* result) {
    if(result == NULL || v1 == NULL || v2 == NULL || v1->data == NULL || v2->data == NULL || v1->rows != 1 || v2->cols != 1 || v1->cols != v2->rows) {return 0;}

    float ret = 0.0f;

    for(size_t i = 0; i < v1->cols; i++) {
        ret += v1->data[i] * v2->data[i];
    }
    
    *result = ret;
    return 1;
}

// ---- MATRIX MULTIPlY ----

int matmul_ijk(matrix* dest, const matrix *a, const matrix *b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->rows || dest->rows != a->rows || dest->cols != b->cols) {return 0;}
    if(dest == a || dest == b) {return 0;}

    for(size_t i = 0; i < dest->rows; i++) {
        for(size_t j = 0; j < dest->cols; j++) {
            for(size_t k = 0; k < a->cols; k++) {
                dest->data[i * dest->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }

    return 1;
}

int matmul_ikj(matrix* dest, const matrix *a, const matrix *b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->rows || dest->rows != a->rows || dest->cols != b->cols) {return 0;}
    if(dest == a || dest == b) {return 0;}

    for(size_t i = 0; i < dest->rows; i++) {
        for(size_t k = 0; k < a->cols; k++) {
            for(size_t j = 0; j < dest->cols; j++) {
                dest->data[i * dest->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }

    return 1;
}

int matmul_jik(matrix* dest, const matrix *a, const matrix *b) {
    if(dest == NULL || dest->data == NULL || a == NULL || b == NULL || a->data == NULL || b->data == NULL) {return 0;}
    if(a->cols != b->rows || dest->rows != a->rows || dest->cols != b->cols) {return 0;}
    if(dest == a || dest == b) {return 0;}

    for(size_t j = 0; j < dest->cols; j++) {
        for(size_t i = 0; i < dest->rows; i++) {
            for(size_t k = 0; k < a->cols; k++) {
                dest->data[i * dest->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }

    return 1;
}

int matmul_tiled(matrix *dest, const matrix *a, const matrix *b, int T) {
    if(dest == NULL || a == NULL || b == NULL || T <= 0 || dest->data == NULL || a->data == NULL || b->data == NULL){return 0;}
    if(a->cols != b->rows || dest->rows != a->rows || dest->cols != b->cols) {return 0;}
    for(size_t ii = 0; ii < a->rows; ii += T) {
        size_t i_end = (ii + T < a->rows) ? ii + T : a->rows;
        for(size_t jj = 0; jj < b->cols; jj += T) {
            size_t j_end = (jj + T < b->cols) ? jj + T : b->cols;
            for(size_t kk = 0; kk < a->cols; kk += T) {
                size_t k_end = (kk + T < a->cols) ? kk + T : a->cols;
                for(size_t i = ii; i < i_end; i++) {
                    for(size_t k = kk; k < k_end; k++) {
                        for(size_t j = jj; j < j_end; j++) {
                            dest->data[i * dest->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                        }
                    }
                }
            }
        }
    }

    return 1;
}

int matvec(matrix *dest, const matrix *a, const matrix *b) {
    if(dest == NULL || a == NULL || b == NULL || dest->data == NULL || a->data == NULL || b->data == NULL){return 0;}
    if(a->cols != b->rows || dest->rows != a->rows || dest->cols != b->cols || b->cols != 1) {return 0;}

    
    for(size_t i = 0; i < a->rows; i++) {
        float sum = 0.0;
        for(size_t j = 0; j < a->cols; j++) {
            sum += a->data[i * a->cols + j] * b->data[j];
        }
        dest->data[i] = sum;
    }

    return 1;
}

// ---- ML OPS ----

int softmax(matrix* dest, const matrix* a, int axis) {
    if(dest == NULL || a == NULL || dest->data == NULL || a->data == NULL) {return 0;}
    if(dest->rows != a->rows || dest->cols != a->cols) {return 0;}
    
    if(axis == 0) { // ROW-AXIS
        for(size_t i = 0; i < a->rows; i++) {
            float argmax = a->data[i * a->cols];
            for(size_t j = 1; j < a->cols; j++) {
                argmax = fmaxf(argmax, a->data[i * a->cols + j]);
            }
            float argsum = 0.0f;
            for(size_t j = 0; j < a->cols; j++) {
                dest->data[i * dest->cols + j] = expf(a->data[i * a->cols + j] - argmax);
                argsum += dest->data[i * dest->cols + j];
            }
            
            float recip = 1.0f / argsum;
            for(size_t j = 0; j < a->cols; j++) {
                dest->data[i * dest->cols + j] *= recip;
            }
        }
    } else if(axis == 1) {
        float* maxes = malloc(sizeof(float) * a->cols);
        float* sums = malloc(sizeof(float) * a->cols);

        if(maxes == NULL || sums == NULL) {
            free(maxes);
            free(sums);
            return 0;
        }

        for (size_t j = 0; j < a->cols; ++j) {
            maxes[j] = a->data[j];
            sums[j] = 0.0f;
        }

        for(size_t i = 0; i < a->rows; i++) {
            for(size_t j = 0; j < a->cols; j++) {
                maxes[j] = fmaxf(maxes[j], a->data[i * a->cols + j]);
            }
        }

        for(size_t i = 0; i < a->rows; i++) {
            for(size_t j = 0; j < a->cols; j++) {
                dest->data[i * dest->cols + j] = expf(a->data[i * a->cols + j] - maxes[j]);
                sums[j] += dest->data[i * dest->cols + j];
            }
        }
        
        for (size_t j = 0; j < a->cols; ++j) {
            sums[j] = 1.0f / sums[j];
        }
        
        for(size_t i = 0; i < a->rows; i++) {
            for(size_t j = 0; j < a->cols; j++) {
                dest->data[i * dest->cols + j] *= sums[j];
            }
        }

        free(maxes);
        free(sums);
    }
    

    return 1;
}

int layernorm(matrix *dest, const matrix *a, const float* gamma, const size_t gamma_length, const float* beta, const size_t beta_length, const float eps) {
    if(dest == NULL || gamma == NULL || beta == NULL || a == NULL || dest->data == NULL || a->data == NULL) {return 0;}
    if(gamma_length != a->cols || beta_length != a->cols || eps < 0 || !isfinite(eps) || dest->rows != a->rows || dest->cols != a->cols) {return 0;}

    for(size_t i = 0; i < a->rows; i++) {
        float mean = 0.0f;
        for(size_t j = 0; j < a->cols; j++) {
            mean += a->data[i * a->cols + j];
        }

        mean /= a->cols;

        float variance = 0.0f;
        for(size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] = a->data[i * a->cols + j] - mean;
            variance += powf(a->data[i * a->cols + j] - mean, 2);
        }

        variance /= a->cols;
        
        for(size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] /= sqrtf(variance + eps);
            dest->data[i * dest->cols + j] = gamma[j] * dest->data[i * dest->cols + j] + beta[j];
        }
    }

    return 1;
}

int rmsnorm(matrix *dest, const matrix* a, const float* gamma, const size_t gamma_length, const float eps) {
    if(dest == NULL || dest->data == NULL || a == NULL || a->data == NULL || gamma == NULL) {return 0;}
    if(dest->cols != a->cols || dest->rows != a->rows || gamma_length != a->cols || eps < 0 || !isfinite(eps)) {return 0;}

    for(size_t i = 0; i < a->rows; i++) {
        float r = 0.0f;
        for(size_t j = 0; j < a->cols; j++) {
            r += powf(a->data[i * a->cols + j], 2);
        }

        r /= a->cols;
        r += eps;
        r = sqrtf(r);

        for(size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] = gamma[j] * a->data[i * a->cols + j] / r;
        }
    }

    return 1;
}

int silu(matrix *dest, matrix *a) {
    if(dest == NULL || a == NULL || dest->data == NULL || a->data == NULL) {return 0;}
    if(dest->cols != a->cols || dest->rows != a->rows) {return 0;}

    for(size_t i = 0; i < dest->rows; i++) {
        for(size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] = a->data[i * dest->cols + j] / (1 + expf(-a->data[i * dest->cols + j]));
        }
    }

    return 1;
}

// ---- TRANSPOSE ----

int transpose(matrix *dest, const matrix *src) {
    if(dest == NULL || src == NULL || dest->data == NULL || src->data == NULL) {return 0;}
    if(dest->rows != src->cols || dest->cols != src->rows) {return 0;}

    for(size_t i = 0; i < dest->rows; i++) {
        for(size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] = src->data[j * dest->rows + i];
        }
    }

    return 1;
}

int matrix_compare(const matrix* lhs, const matrix* rhs) {
    if(lhs == NULL || rhs == NULL || lhs->data == NULL || rhs->data == NULL) {return 0;}
    if(lhs->rows != rhs->rows || lhs->cols != rhs->cols) {return 0;}

    for(size_t i = 0; i < lhs->rows; i++) {
        for(size_t j = 0; j < lhs->cols; j++) {
            if(lhs->data[i * lhs->cols + j] != rhs->data[i * rhs->cols + j]) {
                return 0;
            }
        }
    }
    return 1;
}


