#ifndef FILTERS_H
#define FILTERS_H 1

#include <stdlib.h>

void highcut(float fhi, int nphi, int phase, int n1, float *data, float *newdata, float *tempdata);
void lowcut(float flo, int nplo, int phase, int n1, float *data, float *newdata, float *tempdata);

#endif
