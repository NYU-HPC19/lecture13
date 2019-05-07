#include <stdio.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  VTUData vtk_data;

  // Add points
  vtk_data.coord = {1,0,0, 0,1,0, 0,0,1};

  // Add triangle
  vtk_data.connect = {0,1,2}; // connect vertices
  vtk_data.offset.push_back(vtk_data.connect.size());
  vtk_data.types.push_back(5); // cell-type = triangle

  // Write to file
  vtk_data.WriteVTK("triangle");

  MPI_Finalize();
  return 0;
}

