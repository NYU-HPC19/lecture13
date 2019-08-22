#include <stdio.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  long Nt = 10, Np = 10;

  VTUData vtk_data;
  vtk_data.coord.resize(Nt * Np * 3);
  vtk_data.value.resize(Nt * Np * 1);
  for (long i = 0; i < Nt; i++) {
    for (long j = 0; j < Np; j++) {
      double R = 2 + cos(j*2*M_PI/Np);
      double X = R * cos(i*2*M_PI/Nt);
      double Y = R * sin(i*2*M_PI/Nt);
      double Z = sin(j*2*M_PI/Np);
      vtk_data.coord[(i*Np+j)*3+0] = X;
      vtk_data.coord[(i*Np+j)*3+1] = Y;
      vtk_data.coord[(i*Np+j)*3+2] = Z;
      vtk_data.value[(i*Np+j)] = X+Y*Y+Z*Z*Z;
    }
  }

  vtk_data.connect.resize(Nt * Np * 2 * 3);
  vtk_data.offset.resize(Nt * Np * 2);
  vtk_data.types.resize(Nt * Np * 2);
  for (long i = 0; i < Nt; i++) {
    for (long j = 0; j < Np; j++) {
      long i1 = (i+1)%Nt;
      long j1 = (j+1)%Np;

      // triangle-1
      vtk_data.connect[(i*Np+j)*6+0] = i*Np+j;
      vtk_data.connect[(i*Np+j)*6+1] = i*Np+j1;
      vtk_data.connect[(i*Np+j)*6+2] = i1*Np+j;
      vtk_data.offset[(i*Np+j)*2+0] = (i*Np+j)*6+3;
      vtk_data.types[(i*Np+j)*2+0] = 5;

      // triangle-2
      vtk_data.connect[(i*Np+j)*6+3] = i1*Np+j1;
      vtk_data.connect[(i*Np+j)*6+4] = i*Np+j1;
      vtk_data.connect[(i*Np+j)*6+5] = i1*Np+j;
      vtk_data.offset[(i*Np+j)*2+1] = (i*Np+j)*6+6;
      vtk_data.types[(i*Np+j)*2+1] = 5;
    }
  }

  // Write to file
  vtk_data.WriteVTK("torus");

  MPI_Finalize();
  return 0;
}

