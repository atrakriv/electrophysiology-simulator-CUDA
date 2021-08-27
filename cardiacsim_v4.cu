/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
using namespace std;

#define BS 32

// Utilities
// 

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            cerr << "ERROR: Bad call to gettimeofday" << endl;
            return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
   double **E;
   int nx=n, ny=m;
   E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
   assert(E);
   int j;
   for(j=0;j<ny;j++) 
     E[j] = (double*)(E+ny) + j*nx;
   return(E);
}
    
// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
 double stats(double *E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
	   l2norm += E[j*(n+2)+i]*E[j*(n+2)+i];
	   if (E[j*(n+2)+i] > mx)
	       mx = E[j*(n+2)+i];
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }

// External functions
extern "C" {
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);

__global__ void Solve(double* e,  double* e_prev, double* r,
                      const double alpha, const int n, const int m, const double kk,
                      const double dt, const double a, const double epsilon,
                      const double M1,const double  M2, const double b)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (row+1)*(n+2)+(column+1);

  __shared__ double tile[BS + 2 ][BS + 2 ];

  //local indices
  int tile_col = threadIdx.x + 1;
  int tile_row = threadIdx.y + 1;


  // Mirroring
  if (row == 0 && column<n)
    e_prev[idx - (n+2)] = e_prev[idx + (n+2)];
  if (row == (n-1) && column<n)
    e_prev[idx + (n+2)] = e_prev[idx - (n+2)];
  if (column == 0 && row<n)
    e_prev[idx - 1] = e_prev[idx + 1];
  if (column == (n-1) && row<n)
    e_prev[idx + 1] = e_prev[idx - 1];
  __syncthreads();


  //interior points
  double center = e_prev[idx];
  tile[tile_row][tile_col] = center;
  __syncthreads();

  // gost of tile
  if(tile_row == 1)
    tile[0][tile_col] = e_prev[idx - (n+2)] ;
  else if (tile_row == BS)
    tile[BS+1][tile_col] = e_prev[idx + (n+2)] ;
  if(tile_col == 1)
    tile[tile_row][0] = e_prev[idx - 1] ;
  else if (tile_col == BS)
    tile[tile_row][BS+1] = e_prev[idx + 1] ;
  __syncthreads();


  if (column<n && row<n)
  {
    e[idx] = center + alpha*(tile[tile_row][tile_col+1] + tile[tile_row][tile_col-1] - 4*center + tile[tile_row+1][tile_col] + tile[tile_row-1][tile_col]);
    e[idx] = e[idx] -dt*(kk* e[idx]*(e[idx] - a)*(e[idx]-1)+ e[idx] *r[idx]);
    r[idx] = r[idx] + dt*(epsilon+M1* r[idx]/( e[idx]+M2))*(-r[idx]-kk* e[idx]*(e[idx]-b-1));
  }

}


void simulate (double* d_E,  double* d_E_prev, double* d_R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b)
{ 
    /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */

  int s2 = BS;
  dim3 DimGrid2(ceil((double)m/s2),ceil((double)n/s2), 1); 
  dim3 DimBlock2(s2, s2, 1);
  Solve<<<DimGrid2, DimBlock2>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
}

// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */

  
  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
  
  double T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;
  int num_threads=1; 

  cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads);
  m = n;  
  int num_bytes = (m+2)*(n+2)* sizeof(double);
 

  // pointers to host & device arrays
  double *d_E=0 , *d_E_prev=0 , *d_R =0;
  double *E=0, *E_prev=0, *R =0;

  // malloc a host array
  E = (double*)malloc(num_bytes);
  E_prev = (double*)malloc(num_bytes);
  R = (double*)malloc(num_bytes);
  

  cudaMalloc((void**)&d_E, num_bytes);
  cudaMalloc((void**)&d_E_prev, num_bytes);
  cudaMalloc((void**)&d_R, num_bytes);
  
  
  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j*(n+2)+i] = R[j*(n+2)+i] = 0;
  
  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j*(n+2)+i] = 1.0;
  
  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j*(n+2)+i] = 1.0;

  cudaMemcpy(d_E_prev, E_prev, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, num_bytes, cudaMemcpyHostToDevice);

  /*for(j=0; j<m+2 ; ++j)
  {
    for (i=0; i<n+2 ; i++)
      printf("%.0f ", E_prev[j*(n+2)+i]);
    cout<<'\n';
  }*/

  cout<<'\n';
  
  double dx = 1.0/n;

  // For time integration, these values shouldn't change 
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  cout << "Grid Size       : " << n << endl; 
  cout << "Duration of Sim : " << T << endl; 
  cout << "Time step dt    : " << dt << endl; 
  cout << "Process geometry: " << px << " x " << py << endl;
  if (no_comm)
    cout << "Communication   : DISABLED" << endl;
  
  cout << endl;
  
  // Start the timer
  double t0 = getTime();
  
 
  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;
  
  while (t<T) {
    
    t += dt;
    niter++;
 
    simulate(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b); 
    
    //swap current E with previous E
    double *tmp = d_E; d_E = d_E_prev; d_E_prev = tmp;
    
    /*if (plot_freq){
      int k = (int)(t/plot_freq);
      if ((t - k * plot_freq) < dt){
	splot(E,t,niter,m+2,n+2);
      }
    }*/
  }

  cudaMemcpy(E, d_E, num_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(E_prev, d_E_prev, num_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(R, d_R, num_bytes, cudaMemcpyDeviceToHost);

  double time_elapsed = getTime() - t0;

  double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
  double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;

  cout << "Number of Iterations        : " << niter << endl;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
  cout << "Sustained Gflops Rate       : " << Gflops << endl; 
  cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl; 

  double mx;
  double l2norm = stats(E_prev,m,n,&mx);
  cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

  if (plot_freq){
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }
  
  //cout << "E(300,300): " << E[300*(n+2)+300] << endl;


  // deallocate memory
  free (E); free (E_prev); free (R);
  //free (d_E); free (d_E_prev); free (d_R);  --> segmentation fault !?
  
  return 0;
}
