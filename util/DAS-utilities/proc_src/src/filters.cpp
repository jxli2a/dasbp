/*****************************************************************************
*                           Subroutine highcut                               *
******************************************************************************
*                                                                            *
* highcut                                                                    *
*                                                                            *
*  DESCRIPTION                                                               *
*  Butterworth highcut (lowpass) filter                                      *;
*                                                                            
*  USAGE                                                                     *
*  void highcut(float fhi,int nphi,int phase,float *data,float *newdata,     *
*		float *tempdata, int n1)                                             *
******************************************************************************  
*  INPUT PARAMETERS                                                          *
*  float fhi           cutoff frequency;                                     *
*  int   nphi          number of poles;                                      *
*  int   phase         0=zero phase  1=min phase                             *
*  int   n1            length of data to be filtered                         *
*  float data[n1]      data to be filtered                                   *
*  float newdata[n1]   workspace                                             *
*  float tempdata[n1]  more workspace                                        *
******************************************************************************  
*  KEYWORDS	bandpass filter butterworth high-pass low-pass                   *
*                                                                            *
*  SEE ALSO                                                                  *
*	 Lpfilt Bandpass                                                         *
*                                                                            *
* end of self documentation                                                  *
******************************************************************************
*  Author - Dave Hale                                                        *
*  EDIT HISTORY                                                              *
*       9-30-85	    stew    Convex version                                   *
*       4-30-91	    steve   Saw version with no AP calls                     *
*       8-01-91	    lin	    Fix the bug and change the wrong comments        *
*      14-09-99     james   Back? into C                                     *
*                           fixed various bugs that had slipped in           *
*			                stripped out highcut/lowcut subroutines          *
*                                                                            *
*  Technical reference: Oppenheim, A.V., and Schafer, R.W., 1975,            *
*	 Digital signal processing, Prentice-Hall Inc.                           *
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "filters.h"

#ifndef pi
#define pi 3.1415926536
#endif

#define B(i,j) b[(j)*nb+(i)]
#define D(i,j) d[(j)*nd+(i)]
#define MOD(i,j) (i-j*(i/j))

void highcut(float fhi, int nphi, int phase, int n1, float *data, float *newdata, float *tempdata)
{
	int nodd,nb,j;
	float *b;
	float a,aa,aap4,dtheta,theta0;
	double c;

	int ihi,i1; 

	nodd = MOD(nphi,2);
	if (phase == 0)	{
		if (nodd != 0) nphi = (nphi+1)/2;
		else nphi = nphi/2;
		nodd = MOD(nphi,2); 
	}
	nb = (nphi+1)/2; 
	// b=(float*) malloc(nb*5*sizeof(float));
	b = new float[nb*5];

	a = 2.0*tan(pi*fhi);  /* radius of poles in s-plane */ 
	aa = a*a;
	aap4 = aa+4.0;
	dtheta = pi/nphi;             /*  angular separation of poles  */ 
	if (nodd != 0) theta0 = 0.0;  /*  pole closest to real s axis */
	else theta0 = dtheta/2.0; 
	if (nodd != 0) {
		B(0,0) = a/(a+2.0);
		B(0,1) = B(0,0);
		B(0,2) = 0.0;
		B(0,3) = (a-2.0)/(a+2.0);
		B(0,4) = 0.0;
	}
	for (j=nodd;j<nb;j++) {
		c = 4.0*a*cos(theta0+j*dtheta);
		B(j,0) = aa/(aap4+c);
		B(j,1) = 2.0*B(j,0);
		B(j,2) = B(j,0);
		B(j,3) = (2.0*aa-8.0)/(aap4+c);
		B(j,4) = (aap4-c)/(aap4+c);
	}

	for (ihi=0;ihi<nb;ihi++) {
		for (i1=2;i1<n1;i1++)
			newdata[i1] = B(ihi,0)*data[i1]+B(ihi,1)*data[i1-1]+B(ihi,2)*
			data[i1-2]-B(ihi,3)*newdata[i1-1]-B(ihi,4)*newdata[i1-2];
		for (i1=0;i1<n1;i1++) data[i1] = newdata[i1];
	}

	if (phase == 0) {   /* highcut again in reverse */
		for (i1=2;i1<n1;i1++)
			tempdata[i1] = data[n1+1-i1];
		for (ihi=0;ihi<nb;ihi++) {
			for (i1=2;i1<n1;i1++)
				newdata[i1] = B(ihi,0)*tempdata[i1] + 
	              B(ihi,1)*tempdata[i1-1] + B(ihi,2)*tempdata[i1-2] - 
	              B(ihi,3)* newdata[i1-1] - B(ihi,4)* newdata[i1-2];
			for (i1=0;i1<n1;i1++) tempdata[i1] = newdata[i1];
		}
		for (i1=2;i1<n1;i1++) data[i1] = tempdata[n1+1-i1];
	}

	delete [] b;
	return;
}

/*****************************************************************************
*                           Subroutine lowcut                                *
******************************************************************************
*                                                                            *
* lowcut                                                                     *
*                                                                            *
*  DESCRIPTION                                                               *
*  Butterworth lowcut (highpass) filter                                      *
*                                                                            *
*  USAGE                                                                     *
*  void lowcut(float fhi,int nphi,int phase,int n1,float *data,              *
*       float *newdata, float *tempdata)                                     *
******************************************************************************  
*  INPUT PARAMETERS                                                          *
*  float flo           cutoff frequency;                                     *
*  int   nplo          number of poles;                                      *
*  int   phase         0=zero phase  1=min phase                             *
*  int   n1            length of data to be filtered                         *
*  float data[n1]      data to be filtered                                   *
*  float newdata[n1]   workspace                                             *
*  float tempdata[n1]  more workspace                                        *
******************************************************************************  
*  KEYWORDS	bandpass filter butterworth high-pass low-pass                   *
*                                                                            *
*  SEE ALSO                                                                  *
*	 Lpfilt Bandpass                                                         *
*                                                                            *
* end of self documentation                                                  *
******************************************************************************
*  Author - Dave Hale                                                        *
*  EDIT HISTORY                                                              *
*       9-30-85	    stew    Convex version                                   *
*       4-30-91	    steve   Saw version with no AP calls                     *
*       8-01-91	    lin	    Fix the bug and change the wrong comments        *
*      14-09-99     james   Back? into C                                     *
*                           fixed various bugs that had slipped in           *
*			                stripped out highcut/lowcut subroutines          *
*                                                                            *
*  Technical reference: Oppenheim, A.V., and Schafer, R.W., 1975,            *
*	 Digital signal processing, Prentice-Hall Inc.                           *
*****************************************************************************/

void lowcut(float flo, int nplo, int phase, int n1, float *data, float *newdata, float *tempdata)
{
	int nodd,nd,j;
	float *d;
	float a,aa,aap4,dtheta,theta0,b1,b2,b3,den,e,ee,fno2;
	double c;
	int ilo,i1; 

	nodd = MOD(nplo,2);

	/* compute lowcut filter coefficients if required by transforming a 
	highcut filter with cutoff at half Nyquist */
	if (phase == 0) {
		if (nodd != 0) nplo = (nplo+1)/2;
		else nplo = nplo/2;
		nodd = MOD(nplo,2);
	}
	nd = (nplo+1)/2;
	// d=(float*) malloc(nd*5*sizeof(float));
	d = new float[nd*5];
 
	fno2 = 0.25;   /*   Nyquist frequency over two?? */
	a = 2.*sin(pi*fno2)/cos(pi*fno2); aa = a*a; aap4 = aa+4;
	e = -cos(pi*(flo+fno2))/cos(pi*(flo-fno2)); ee = e*e;
	dtheta = pi/nplo;		        /*  angular separation of poles  */
	if (nodd != 0) theta0 = 0;		/*  pole closest to real s axis */
	else theta0 = dtheta/2;
	if (nodd != 0) {
		b1 = a/(a+2);
		b2 = (a-2)/(a+2);
		den = 1.0-b2*e;
		D(0,0) = b1*(1.-e)/den;
		D(0,1) = -D(0,0);
		D(0,2) = 0.;
		D(0,3) = (e-b2)/den;
		D(0,4) = 0.;
	} 
	for (j=nodd;j<nd;j++) {
		c = 4.*a*cos(theta0+j*dtheta);
		b1 = aa/(aap4+c);
		b2 = (2.*aa-8.)/(aap4+c);
		b3 = (aap4-c)/(aap4+c);
		den = 1.-b2*e+b3*ee;
		D(j,0) = b1*(1.-e)*(1.-e)/den;
		D(j,1) = -2.*D(j,0);
		D(j,2) = D(j,0);
		D(j,3) = (2.*e*(1.+b3)-b2*(1.+ee))/den;
		D(j,4) = (ee-b2*e+b3)/den;
	}

	/* lowcut filter */
	for (ilo=0;ilo<nd;ilo++) {
		for (i1=2;i1<n1;i1++)
			newdata[i1] = D(ilo,0)*data[i1]+D(ilo,1)* data[i1-1]+D(ilo,2)*
				data[i1-2]-D(ilo,3)*newdata[i1-1]- D(ilo,4)*newdata[i1-2];
		for (i1=0;i1<n1;i1++) data[i1] = newdata[i1];
	}
    
	/* lowcut again in reverse */
	if (phase == 0) {
		for (i1=2;i1<n1;i1++)
			tempdata[i1] = data[n1+1-i1];
		for (ilo=0;ilo<nd;ilo++) {
			for (i1=2;i1<n1;i1++)
				newdata[i1]=D(ilo,0)*tempdata[i1]+D(ilo,1)*tempdata[i1-1]+
					D(ilo,2)*tempdata[i1-2]-D(ilo,3)*newdata[i1-1]-
					D(ilo,4)*newdata[i1-2];
			for (i1=0;i1<n1;i1++) tempdata[i1]=newdata[i1];
		}
		for (i1=2;i1<n1;i1++)
			data[i1] = tempdata[n1+1-i1];
	}

	delete [] d;
	return;
}