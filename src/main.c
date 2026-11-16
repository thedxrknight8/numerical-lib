#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <stdalign.h>
#include <time.h>

const int ROWS_A = 1024;
const int COLS_A = 1024;

const int ROWS_B = 1024;
const int COLS_B = 1024;

const int ROWS_D = 1024;
const int COLS_D = 1024;

int main(void) {

    matrix a_mat = {0};
    matrix b_mat = {0};
    matrix dest = {0};
    matrix dest2 = {0};

    matrix_init(&a_mat, ROWS_A, COLS_A);
    matrix_init(&b_mat, ROWS_B, COLS_B);
    matrix_init(&dest, ROWS_D, COLS_D);
    matrix_init(&dest2, ROWS_D, COLS_D);

    clock_t start, end;

    matrix_fill_random(&a_mat);
    matrix_fill_random(&b_mat);
    matrix_fill_zeroes(&dest);
    matrix_fill_zeroes(&dest2);

    matrix_add_ij(&dest, &a_mat, &b_mat);

    // // ---- MATRIX ADD TESTS ----
    // start = clock();
    // matrix_add_ij(&dest, &a_mat, &b_mat);
    // end = clock();
    // double matrix_add_ij_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

    // printf("[matrix_add_ij] - ROWS = %d, COLS = %d - %f seconds\n", ROWS_A, COLS_A, matrix_add_ij_clock);
    // fflush(stdout);

    // start = clock();
    // matrix_add_ji(&dest, &a_mat, &b_mat);
    // end = clock();
    // double matrix_add_ji_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

    // printf("[matrix_add_ji] - ROWS = %d, COLS = %d - %f seconds\n\n", ROWS_A, COLS_A, matrix_add_ji_clock);
    // fflush(stdout);

    // ---- MATRIX MULTIPLY TESTS ----
    
    for(int i = 0; i < 5; i++) {
        matrix_fill_zeroes(&dest);
        start = clock();
        matmul_ijk(&dest, &a_mat, &b_mat);
        end = clock();
        double matmul_ijk_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("RUN #%d - [matmul_ijk] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, matmul_ijk_clock);
        fflush(stdout);
    }
    printf("\n");
    
    for(int i = 0; i < 5; i++) {
        matrix_fill_zeroes(&dest);
        start = clock();
        matmul_ikj(&dest, &a_mat, &b_mat);
        end = clock();
        double matmul_ikj_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("RUN #%d - [matmul_ikj] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, matmul_ikj_clock);
        fflush(stdout);
    }
    printf("\n");

    int tile_sizes[] = {32};
    for(int j = 0; j < 1; j++) {
        for(int i = 0; i < 5; i++) {
            matrix_fill_zeroes(&dest);
            start = clock();
            matmul_tiled(&dest, &a_mat, &b_mat, tile_sizes[j]);
            end = clock();
            double matmul_tiled_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

            printf("RUN %d [matmul_tiled] - T=%d - ROWS = %d, COLS = %d - %f seconds\n", i + 1, tile_sizes[j], ROWS_D, COLS_D, matmul_tiled_clock);
            fflush(stdout);
        }
        printf("\n");
    }
    
    for(int i = 0; i < 5; i++) {
        matrix_fill_zeroes(&dest);
        start = clock();
        matmul_jik(&dest, &a_mat, &b_mat);
        end = clock();
        double matmul_jik_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("RUN #%d - [matmul_jik] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, matmul_jik_clock);
        fflush(stdout);
    }
    printf("\n");

    for(int i = 0; i < 5; i++) {
        matrix_fill_zeroes(&dest);
        start = clock();
        matvec(&dest, &a_mat, &b_mat);
        end = clock();
        double matmul_tiled_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("RUN #%d - [matvec] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, matmul_tiled_clock);
        fflush(stdout);
    }
    printf("\n");

    

    // for(int i = 0; i < 5; i++) {
    //     matrix_fill_zeroes(&dest);
    //     start = clock();
    //     softmax(&dest, &a_mat, 1);
    //     end = clock();
    //     double softmax_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

    //     printf("RUN %d [softmax] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, softmax_clock);
    //     fflush(stdout);
    // }

    // float gamma[COLS_D];
    // float beta[COLS_D];

    // for(int i = 0; i < COLS_D; i++) {
    //     gamma[i] = 1.0f;
    //     beta[i] = 0.0f;
    // }

    // layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);
    // layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);
    // layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);
    // layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);
    // layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);

    // for(int i = 0; i < 100; i++) {
    //     matrix_fill_zeroes(&dest);
    //     start = clock();
    //     layernorm(&dest, &a_mat, gamma, COLS_A, beta, COLS_A, 1e-5);
    //     end = clock();
    //     double layernnorm_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

    //     printf("RUN %d [layernorm] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, layernnorm_clock);
    //     fflush(stdout);
    // }

    // float gamma[COLS_D];
    // for(int i = 0; i < COLS_D; i++) {
    //     gamma[i] = 1.0f;
    // }

    // rmsnorm(&dest, &a_mat, gamma, COLS_D, 1e-5);
    // rmsnorm(&dest, &a_mat, gamma, COLS_D, 1e-5);
    // rmsnorm(&dest, &a_mat, gamma, COLS_D, 1e-5);
    // rmsnorm(&dest, &a_mat, gamma, COLS_D, 1e-5);
    // rmsnorm(&dest, &a_mat, gamma, COLS_D, 1e-5);

    // for(int i = 0; i < 100; i++) {
    //     matrix_fill_zeroes(&dest);
    //     start = clock();
    //     rmsnorm(&dest, &a_mat, gamma, COLS_A, 1e-5);
    //     end = clock();
    //     double rmsnorm_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

    //     printf("RUN %d [rmsnorm] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, rmsnorm_clock);
    //     fflush(stdout);
    // }

    for(int i = 0; i < 100; i++) {
        matrix_fill_zeroes(&dest);
        start = clock();
        silu(&dest, &a_mat);
        end = clock();
        double silu_clock = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("RUN %d [silu] - ROWS = %d, COLS = %d - %f seconds\n", i + 1, ROWS_D, COLS_D, silu_clock);
        fflush(stdout);
    }

    // matrix_destroy(&a_mat);
    matrix_destroy(&b_mat);
    matrix_destroy(&dest);
    matrix_destroy(&dest2);

    return 0;
}