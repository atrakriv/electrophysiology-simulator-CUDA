#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            printf("ERROR: Bad call to gettimeofday\n");
            return(-1);
    }
    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}


__global__ void kernel(double* e,  double* e_prev, double* r,
                      const double alpha, const int n, const int m, const double kk,
                      const double dt, const double a, const double epsilon,
                      const double M1,const double  M2, const double b) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (row+1)*(n+2)+(column+1);

    if (row == 0 && column<n)
        e_prev[idx - (n+2)] = e_prev[idx + (n+2)];
    if (row == (n-1) && column<n)
        e_prev[idx + (n+2)] = e_prev[idx - (n+2)];
    if (column == 0 && row<n)
        e_prev[idx - 1] = e_prev[idx + 1];
    if (column == (n-1) && row<n)
        e_prev[idx + 1] = e_prev[idx - 1];

    if (column<n && row<n) {
        // order matters!
        e[idx] = e_prev[idx]+alpha*(e_prev[idx+1] + e_prev[idx-1] - 4*e_prev[idx] + e_prev[idx+(n+2)] + e_prev[idx-(n+2)]);
        e[idx] = e[idx] -dt*(kk* e[idx]*(e[idx] - a)*(e[idx]-1)+ e[idx] *r[idx]);
        r[idx] = r[idx] + dt*(epsilon+M1* r[idx]/( e[idx]+M2))*(-r[idx]-kk* e[idx]*(e[idx]-b-1));
    }
}


void simulate (double* d_E,  double* d_E_prev, double* d_R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b) {

    int s2 = 32;
    dim3 DimGrid2(ceil((double)m/s2),ceil((double)n/s2), 1); 
    dim3 DimBlock2(s2, s2, 1);

    kernel<<<DimGrid2, DimBlock2>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b); 
}


int main() {

    int m = 200, n=200;
    double T=1000.0;

    const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
   
    double *E, *E_prev, *R;
    double *d_E, *d_E_prev, *d_R;
   
    int num_bytes = (m+2)*(n+2)*sizeof(double);

    E = (double*)malloc(num_bytes);
    E_prev = (double*)malloc(num_bytes);
    R = (double*)malloc(num_bytes);
    
    cudaMalloc((void**)&d_E, num_bytes);
    cudaMalloc((void**)&d_E_prev, num_bytes);
    cudaMalloc((void**)&d_R, num_bytes);

    // initialization
    for (int j=1; j<=m; j++)
        for (int i=1; i<=n; i++)
            E_prev[j*(n+2)+i] = R[j*(n+2)+i] = 0;
    // initialization
    for (int j=1; j<=m; j++)
        for (int i=n/2+1; i<=n; i++)
            E_prev[j*(n+2)+i] = 1.0;
    // initialization
    for (int j=m/2+1; j<=m; j++)
        for (int i=1; i<=n; i++)
            R[j*(n+2)+i] = 1.0;
    
    cudaMemcpy(d_E_prev, E_prev, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, num_bytes, cudaMemcpyHostToDevice);

    double dx = 1.0/n;
    double t = 0.0;
    int cnt=0;

    double rp= kk*(b+1)*(b+1)/4;
    double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
    double dtr=1/(epsilon+((M1/M2)*rp));
    double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
    double alpha = d*dt/(dx*dx);

    double t0 = getTime();

    while (t<T) {
        t += dt;
        cnt++;
        simulate(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
        //swap current E with previous E
        double *tmp = d_E; d_E = d_E_prev; d_E_prev = tmp;
    }

    cudaMemcpy(E, d_E, num_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(E_prev, d_E_prev, num_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(R, d_R, num_bytes, cudaMemcpyDeviceToHost);

    double time_elapsed = getTime() - t0;
    printf("Elapsed Time (sec) %g\n",time_elapsed);

    FILE * fp = fopen("v2.txt","w");
    for (int i=1; i<m+1; i++)
        for (int j=1; j<n+1; j++)
            fprintf(fp,"%f\n", E[i*(n+2)+j]);
    fclose(fp);

    free(E); free(E_prev); free(R);
    cudaFree(d_E); cudaFree(d_E_prev); cudaFree(d_R);
 
return 0;

}
