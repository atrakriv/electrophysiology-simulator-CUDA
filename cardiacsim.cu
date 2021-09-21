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
 double stats(double **E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
	   l2norm += E[j][i]*E[j][i];
	   if (E[j][i] > mx)
	       mx = E[j][i];
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }


void simulate (double** E,  double** E_prev,double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b)
{
  int i, j; 
 
  for (j=1; j<=m; j++) 
      E_prev[j][0] = E_prev[j][2];
  for (j=1; j<=m; j++) 
      E_prev[j][n+1] = E_prev[j][n-1];
    
  for (i=1; i<=n; i++) 
      E_prev[0][i] = E_prev[2][i];
  for (i=1; i<=n; i++) 
      E_prev[m+1][i] = E_prev[m-1][i];
    
    // Solve for the excitation, the PDE
    for (j=1; j<=m; j++){
        for (i=1; i<=n; i++) {
	          E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);
        }
    }
    

  for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	        E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);
  }
    
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	        R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));
    }
    
}

// Main program
int main (int argc, char** argv)
{
  double **E, **R, **E_prev;
  
  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
  
  double T=1000.0;
  int m=200,n=200;

  m = n;  
  E = alloc2D(m+2,n+2);
  E_prev = alloc2D(m+2,n+2);
  R = alloc2D(m+2,n+2);
  
  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j][i] = R[j][i] = 0;
  
  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j][i] = 1.0;
  
  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j][i] = 1.0;
  
  double dx = 1.0/n;

  // For time integration, these values shouldn't change 
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  
  double t0 = getTime();
  double t = 0.0;
  int niter=0;

  while (t<T) {
    
    t += dt;
    niter++;
 
    simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b); 
    //swap current E with previous E
    double **tmp = E; E = E_prev; E_prev = tmp;
  }

  double time_elapsed = getTime() - t0;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;

  FILE * fp = fopen("v0.txt","w");
  for (int i=1; i<m+1; i++)
      for (int j=1; j<n+1; j++)
          fprintf(fp,"%f\n", E[i][j]);
  fclose(fp);
  
  free (E);
  free (E_prev);
  free (R);
  
  return 0;
}
