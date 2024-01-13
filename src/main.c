#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "inv_matrix.h"

/* reference: [1] https://mp.weixin.qq.com/s/_nIWlXXyhAKhyTltez72HQ
[2] 黄小平, 王岩. 卡尔曼滤波原理及应用: MATLAB 仿真[M]. 电子工业出版社, 2015.*/

#define ARRAY_SIZE 2000   // sampling times
static double Ts = 0.001; // sampling period
static double t0 = 0.0;   // start time
static double t1 = 2.0;   // end time

// calculate matrix multiplication
void matrix_multi(double *C, double *A, double *B, int rows1, int cols1, int cols2){
    for (int j = 0; j < rows1; j++){
        for (int k = 0; k < cols2; k++){
            *(C + j * cols2 + k) = 0.0;
            for (int g = 0; g < cols1; g++){
                *(C + j * cols2 + k) += *(A + j * cols1 + g) * *(B + g * cols2 + k);
            }
        }
    }
}

// calculate the transpose of matrix
void matrix_transpose(int rows, int cols, double matrix[rows][cols], double result[cols][rows]){
    for (int j = 0; j < rows; j++){
        for (int k = 0; k < cols; k++){
            result[k][j] = matrix[j][k];
        }
    }
}

// define state variable and measurement variable
typedef struct{
    double data[3];
} state_variable;

typedef struct{
    double data[3];
} measurement_variable;

// state transition function
void state_transition_function(const state_variable *state, const double *u, state_variable *new_state){
    new_state->data[0] = state->data[0] + state->data[1] * cos(state->data[2]);
    new_state->data[1] = state->data[1] - *u;
    new_state->data[2] = state->data[2] + pow((*u), 2);
}

// observation function
void observation_function(const state_variable *state, measurement_variable *observation){
    observation->data[0] = state->data[0] * state->data[0];
    observation->data[1] = state->data[1] * state->data[1];
    observation->data[2] = state->data[2] * state->data[2];
}

// random generation of control input
double generate_control_input(){
    static int initialized = 0;
    if (!initialized){
        srand(time(NULL));
        initialized = 1;
    }

    double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    return 0.5 * x;
}

struct _ekf{
    state_variable true_state;                  // true state variable
    state_variable estimated_state;             // initialize estimated state variable
    state_variable predicted_state;             // predicted state variable
    state_variable updated_state;               // updated state variable
    measurement_variable measurement;           // observation variable
    measurement_variable measurement_predicted; // predicted observation variable
    double covariance[3][3];                    // covariance matrix P
    double predicted_covariance[3][3];          // predicted covariance matrix
    double updated_covariance[3][3];            // updated covariance matrix
    double Q[3][3];                             // process noise W covariance matrix Q
    double R[3][3];                             // gaussian white noise V covariance matrix R
    double K[3][3];                             // kalman gain
    double jacobian_F[3][3];                    // jacobian matrice of nonlinear function f on state variable x, Phi
    double jacobian_H[3][3];                    // observation matrix H, i.e., jacobian matrix of observation function h for state variable x

} ekf;

void EKF_init(){
    // initialize state variable and covariance matrix
    for (int j = 0; j < 3; j++){ // initialize true state variable
        ekf.true_state.data[j] = 0;
    }
    for (int j = 0; j < 3; j++){ // initialize estimated state variable
        ekf.estimated_state.data[j] = 0;
    }

    // initialize covariance matrix P
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            if (j == k){
                ekf.covariance[j][k] = 1;
            }
            else{
                ekf.covariance[j][k] = 0;
            }
        }
    }

    // process noise W covariance matrix Q
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            if (j == k){
                ekf.Q[j][k] = 0.1;
            }
            else{
                ekf.Q[j][k] = 0;
            }
        }
    }

    // gaussian white noise V covariance matrix R
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            if (j == k){
                ekf.R[j][k] = 1;
            }
            else{
                ekf.R[j][k] = 0;
            }
        }
    }
}

// extended kalman filter
void EKF_realize(int i){
    // generate randomized control input
    double control = generate_control_input();

    // update true state variable
    state_transition_function(&ekf.true_state, &control, &ekf.true_state);

    // prediction step
    // predicted state variabl is equal to state transition matrix multiplied by estimated state variable
    state_transition_function(&ekf.estimated_state, &control, &ekf.predicted_state);

    // jacobian matrice of nonlinear function f on state variable x, Phi
    ekf.jacobian_F[0][0] = 1;
    ekf.jacobian_F[0][1] = cos(ekf.estimated_state.data[2]);
    ekf.jacobian_F[0][2] = -ekf.estimated_state.data[1] * sin(ekf.estimated_state.data[2]);
    ekf.jacobian_F[1][0] = 0;
    ekf.jacobian_F[1][1] = 1;
    ekf.jacobian_F[1][2] = 0;
    ekf.jacobian_F[2][0] = 0;
    ekf.jacobian_F[2][1] = 2 * control;
    ekf.jacobian_F[2][2] = 1;

    // transpose matrix of state transfer matrix, Phi'
    double jacobian_F_transpose[3][3];
    matrix_transpose(3, 3, ekf.jacobian_F, jacobian_F_transpose);

    // compute jacobian matrix respect to f multiplied by covariance matrix P
    double jacobian_F_covariance[3][3];
    matrix_multi((double *)jacobian_F_covariance, (double *)ekf.jacobian_F, (double *)ekf.covariance, 3, 3, 3);

    // then multiply by transpose of jacobian matrix respect to f
    double temp[3][3];
    matrix_multi((double *)temp, (double *)jacobian_F_covariance, (double *)jacobian_F_transpose, 3, 3, 3);

    /* predicted covariance matrix is equal to jacobian matrix about f multiplied by covariance matrix P multiplied
    by transpose of jacobian matrix about f plus noise variance Q*/
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            ekf.predicted_covariance[j][k] = temp[j][k] + ekf.Q[j][k];
        }
    }

    // update step
    observation_function(&ekf.true_state, &ekf.measurement); // observation value Z of state variable, considering sensor error
    ekf.measurement.data[0] += rand() / (double)RAND_MAX;
    ekf.measurement.data[1] += rand() / (double)RAND_MAX;
    ekf.measurement.data[2] += rand() / (double)RAND_MAX;

    // observation matrix H, i.e., jacobian matrix of observation function h for state variable x
    ekf.jacobian_H[0][0] = 2 * ekf.true_state.data[0];
    ekf.jacobian_H[0][1] = 0;
    ekf.jacobian_H[0][2] = 0;
    ekf.jacobian_H[1][0] = 0;
    ekf.jacobian_H[1][1] = 2 * ekf.true_state.data[1];
    ekf.jacobian_H[1][2] = 0;
    ekf.jacobian_H[2][0] = 0;
    ekf.jacobian_H[2][1] = 0;
    ekf.jacobian_H[2][2] = 2 * ekf.true_state.data[2];

    // transpose matrixof jacobian matrix with respect to observed function h, H'
    double jacobian_H_transpose[3][3];
    matrix_transpose(3, 3, ekf.jacobian_H, jacobian_H_transpose);

    // compute jacobian matrix with respect to observed function h multiplied by predicted covariance matrix
    double jacobian_H_pre_covariance[3][3];
    matrix_multi((double *)jacobian_H_pre_covariance, (double *)ekf.jacobian_H, (double *)ekf.predicted_covariance, 3, 3, 3);

    // then multiply by transpose of jacobian matrix respect to h
    double temp1[3][3];
    matrix_multi((double *)temp1, (double *)jacobian_H_pre_covariance, (double *)jacobian_H_transpose, 3, 3, 3);

    /* temporary matrix S is equal to jacobian matrix about h multiplied by predicted covariance matrix multiplied
    by transpose of jacobian matrix about h plus noise variance R */
    double S[3][3];
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            S[j][k] = temp1[j][k] + ekf.R[j][k];
        }
    }

    double inv_S[3][3]; // compute inverse matrix of temporary matrix S
    inv_matrix(inv_S, S, 3);

    // compute predicted covariance matrix multiplied by transpose matrix of jacobian matrix about h
    double pre_covariance_jacobian_Ht[3][3];
    matrix_multi((double *)pre_covariance_jacobian_Ht, (double *)ekf.predicted_covariance, (double *)jacobian_H_transpose, 3, 3, 3);

    /* kalman gain K is equal to predicted covariance matrix multiplied by transpose matrix of jacobian matrix about h multiplied
    by transpose of temporary matrix */
    matrix_multi((double *)ekf.K, (double *)pre_covariance_jacobian_Ht, (double *)inv_S, 3, 3, 3);

    // observation minus observation of estimated predicted state
    double innovation[3];
    observation_function(&ekf.predicted_state, &ekf.measurement_predicted);
    innovation[0] = ekf.measurement.data[0] - ekf.measurement_predicted.data[0];
    innovation[1] = ekf.measurement.data[1] - ekf.measurement_predicted.data[1];
    innovation[2] = ekf.measurement.data[2] - ekf.measurement_predicted.data[2];

    // compute kalman gain multiplied by observation minus observation of estimated predicted state
    double K_innovation[3];
    matrix_multi((double *)K_innovation, (double *)ekf.K, (double *)innovation, 3, 3, 1);

    // updated state variable is equal to predicted state variable plus kalman gain multiplied by observation minus observation of estimated predicted state
    for (int j = 0; j < 3; j++){
        ekf.updated_state.data[j] = ekf.predicted_state.data[j] + K_innovation[j];
    }

    // compute kalman gain multiplied by jacobian matrix with respect to observed function h
    double K_jacobian_H[3][3];
    matrix_multi((double *)K_jacobian_H, (double *)ekf.K, (double *)ekf.jacobian_H, 3, 3, 3);

    double Identity[3][3] = { // identity matrix I
                             {1, 0, 0},
                             {0, 1, 0},
                             {0, 0, 1}};

    /* identity matrix I minus kalman gain multiplied by jacobian matrix with respect to observed function h*/
    double I_K_jacobian_H[3][3];
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            I_K_jacobian_H[j][k] = Identity[j][k] - K_jacobian_H[j][k];
        }
    }

    /* updated covariance matrix is equal to identity matrix I minus kalman gain K multiplied by jacobian matrixwith respect to
    observed function h, multiplied by predicted covariance matrix*/
    matrix_multi((double *)ekf.updated_covariance, (double *)I_K_jacobian_H, (double *)ekf.predicted_covariance, 3, 3, 3);

    // update filter state variable and covariance matrix
    ekf.estimated_state = ekf.updated_state;
    for (int j = 0; j < 3; j++){
        for (int k = 0; k < 3; k++){
            ekf.covariance[j][k] = ekf.updated_covariance[j][k];
        }
    }

    printf("true state     : ");
    printf("%f, %f, %f\n", ekf.true_state.data[0], ekf.true_state.data[1], ekf.true_state.data[2]);

    printf("filtered state : ");
    printf("%f, %f, %f\n", ekf.estimated_state.data[0], ekf.estimated_state.data[1], ekf.estimated_state.data[2]);

    printf("state error    : ");
    printf("%f, %f, %f\n", ekf.true_state.data[0] - ekf.estimated_state.data[0],
           ekf.true_state.data[1] - ekf.estimated_state.data[1],
           ekf.true_state.data[2] - ekf.estimated_state.data[2]);

    printf("\n");
}

int main(){

    EKF_init(); // initialize controller parameter
    // PLANT_init();       // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++){
        // for (int i = 0; i < 200; i++){
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        EKF_realize(i);
        // PLANT_realize(i);
    }

    // saveArchive();

    return 0;
}
