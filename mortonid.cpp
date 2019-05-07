#include <stdio.h>
#include <string>
#include <mpi.h>
#include "utils.h"

class Particle {
  public:

    Particle() = default;

    Particle(double x, double y, double z) {
      coord[0] = x;
      coord[1] = y;
      coord[2] = z;

      coord[0] -= floor(coord[0]);
      coord[1] -= floor(coord[1]);
      coord[2] -= floor(coord[2]);

      // convert coord to integer coordinates
      // map [0,1] --> {0, 1, ..., 2^20}
      unsigned long int_coord[3];
      constexpr unsigned int MAX_DEPTH = 20;
      constexpr unsigned long s = (1u << MAX_DEPTH);
      int_coord[0] = floor(coord[0] * s);
      int_coord[1] = floor(coord[1] * s);
      int_coord[2] = floor(coord[2] * s);

      mid = 0;
      unsigned long flag1 = 1;
      unsigned long flag3 = 1;
      for (int i = 0; i < MAX_DEPTH; i++) {
        mid += ((int_coord[0] & flag1) != 0) * (flag3<<0);
        mid += ((int_coord[1] & flag1) != 0) * (flag3<<1);
        mid += ((int_coord[2] & flag1) != 0) * (flag3<<2);

        flag1 = flag1 << 1;
        flag3 = flag3 << 3;
      }
    }

    double GetCoord(int i) const { return coord[i]; }
    unsigned long GetMID() const { return mid; }

    bool operator<(const Particle& p) const { return mid < p.mid; }

  private:
    unsigned long mid;
    double coord[3];
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  srand48(rank);

  long N = 10000/np;
  std::vector<Particle> pts(N);
  for (long i = 0; i < N; i++) {
    double X = drand48();
    double Y = drand48();
    double Z = drand48();
    pts[i] = Particle(X,Y,Z);
  }

  { // Write VTK (before)
    VTUData vtk_data;
    for (long i = 0; i < N; i++) {
      vtk_data.coord.push_back(pts[i].GetCoord(0));
      vtk_data.coord.push_back(pts[i].GetCoord(1));
      vtk_data.coord.push_back(pts[i].GetCoord(2));
      vtk_data.value.push_back(pts[i].GetMID());
      vtk_data.connect.push_back(i);
    }
    vtk_data.offset.push_back(vtk_data.connect.size());
    vtk_data.types.push_back(4);
    vtk_data.WriteVTK("before", comm);
  }

  MPI_BitonicSort(pts, comm);

  { // Write VTK (after)
    VTUData vtk_data;
    for (long i = 0; i < N; i++) {
      vtk_data.coord.push_back(pts[i].GetCoord(0));
      vtk_data.coord.push_back(pts[i].GetCoord(1));
      vtk_data.coord.push_back(pts[i].GetCoord(2));
      vtk_data.value.push_back(pts[i].GetMID());
      vtk_data.connect.push_back(i);
    }
    vtk_data.offset.push_back(vtk_data.connect.size());
    vtk_data.types.push_back(4);
    vtk_data.WriteVTK("after", comm);
  }

  MPI_Finalize();
  return 0;
}

