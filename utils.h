#ifndef _UTILS_H_
#define _UTILS_H_

#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>

#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <mpi.h>


class Timer {
  public:

    void tic() {
      t_start = std::chrono::high_resolution_clock::now();
    }

    double toc() {
      auto t_end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};

template <class ValueType> ValueType read_option(const char* option, int argc, char** argv, const char* default_value = nullptr);

template <> std::string read_option<std::string>(const char* option, int argc, char** argv, const char* default_value) {
  for (int i = 0; i < argc - 1; i++) {
    if (!strcmp(argv[i], option)) {
      return std::string(argv[i+1]);
    }
  }
  if (default_value) return std::string(default_value);
  std::cerr<<"Option "<<option<<" was not provided. Exiting...\n";
  exit(1);
}
template <> int read_option<int>(const char* option, int argc, char** argv, const char* default_value) {
  return strtol(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL, 10);
}
template <> long read_option<long>(const char* option, int argc, char** argv, const char* default_value) {
  return strtol(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL, 10);
}
template <> float read_option<float>(const char* option, int argc, char** argv, const char* default_value) {
  return strtod(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL);
}
template <> double read_option<double>(const char* option, int argc, char** argv, const char* default_value) {
  return strtof(read_option<std::string>(option, argc, argv, default_value).c_str(), NULL);
}

/**
 * \class CommDatatype
 * \brief An abstract class used for communicating messages using user-defined
 * datatypes. The user must implement the static member function "value()" that
 * returns the MPI_Datatype corresponding to this user-defined datatype.
 * \author Hari Sundar, hsundar@gmail.com
 */
template <class Type> class CommDatatype {
 public:
  static MPI_Datatype value() {
    static bool first = true;
    static MPI_Datatype datatype;
    if (first) {
      first = false;
      MPI_Type_contiguous(sizeof(Type), MPI_BYTE, &datatype);
      MPI_Type_commit(&datatype);
    }
    return datatype;
  }
};

template <class KeyType> void MPI_BitonicSort(std::vector<KeyType>& A, MPI_Comm comm) {
  int np, rank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);

  long N = A.size();
  MPI_Bcast(&N, 1, MPI_LONG, 0, comm);
  assert(A.size() == N);

  std::vector<KeyType> B(N);
  std::sort(A.begin(), A.end());

  int np_ = 1;
  while (np_ < np) np_ *= 2;
  for (long len = 2; len <= np_; len = len << 1) {
    { // bitonic merge step0
      int offset = rank % len;
      int partner = rank + len - 2*offset - 1;
      if (partner < np) {
        MPI_Sendrecv(&A[0], N, CommDatatype<KeyType>::value(), partner, len,
                     &B[0], N, CommDatatype<KeyType>::value(), partner, len,
                     comm, MPI_STATUS_IGNORE);

        if (rank < partner) {
          for (long l = 0; l < N; l++) {
            if (B[N-1-l] < A[l]) std::swap(B[N-1-l], A[l]);
          }
        } else {
          for (long l = 0; l < N; l++) {
            if (A[N-1-l] < B[l]) std::swap(A[N-1-l], B[l]);
          }
        }
      }
    }
    for (long j = len/2; j > 1; j=j>>1) { // bitonic merge step1
      int offset = rank % j;
      int partner = rank - offset + ((offset+j/2) % j);
      if (partner < np) {
        MPI_Sendrecv(&A[0], N, CommDatatype<KeyType>::value(), partner, len,
                     &B[0], N, CommDatatype<KeyType>::value(), partner, len,
                     comm, MPI_STATUS_IGNORE);

        if (rank < partner) {
          for (long l = 0; l < N; l++) {
            if (B[l] < A[l]) std::swap(B[l], A[l]);
          }
        } else {
          for (long l = 0; l < N; l++) {
            if (A[l] < B[l]) std::swap(A[l], B[l]);
          }
        }
      }
    }
    std::sort(A.begin(), A.end());
  }
}

struct VTUData {
  typedef float VTKReal;

  // Point data
  std::vector<VTKReal> coord;  // always 3D
  std::vector<VTKReal> value;

  // Cell data
  std::vector<int32_t> connect;
  std::vector<int32_t> offset;
  std::vector<uint8_t> types;

  void WriteVTK(const std::string& fname, MPI_Comm comm = MPI_COMM_SELF) const {
    typedef typename VTUData::VTKReal VTKReal;

    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    long value_dof = 0;
    {  // Write vtu file.
      std::ofstream vtufile;
      {  // Open file for writing.
        std::stringstream vtufname;
        vtufname << fname << std::setfill('0') << std::setw(6) << rank << ".vtu";
        vtufile.open(vtufname.str().c_str());
        if (vtufile.fail()) return;
      }
      {  // Write to file.
        long pt_cnt = coord.size() / 3;
        long cell_cnt = types.size();
        value_dof = (pt_cnt ? value.size() / pt_cnt : 0);

        std::vector<int32_t> mpi_rank;
        {  // Set  mpi_rank
          int new_myrank = rank;
          mpi_rank.resize(pt_cnt);
          for (long i = 0; i < mpi_rank.size(); i++) mpi_rank[i] = new_myrank;
        }

        bool isLittleEndian;
        {  // Set isLittleEndian
          uint16_t number = 0x1;
          uint8_t *numPtr = (uint8_t *)&number;
          isLittleEndian = (numPtr[0] == 1);
        }

        long data_size = 0;
        vtufile << "<?xml version=\"1.0\"?>\n";
        vtufile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << (isLittleEndian ? "LittleEndian" : "BigEndian") << "\">\n";
        // ===========================================================================
        vtufile << "  <UnstructuredGrid>\n";
        vtufile << "    <Piece NumberOfPoints=\"" << pt_cnt << "\" NumberOfCells=\"" << cell_cnt << "\">\n";
        //---------------------------------------------------------------------------
        vtufile << "      <Points>\n";
        vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + coord.size() * sizeof(VTKReal);
        vtufile << "      </Points>\n";
        //---------------------------------------------------------------------------
        vtufile << "      <PointData>\n";
        if (value_dof) {  // value
          vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\" format=\"appended\" offset=\"" << data_size << "\" />\n";
          data_size += sizeof(uint32_t) + value.size() * sizeof(VTKReal);
        }
        {  // mpi_rank
          vtufile << "        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\"" << data_size << "\" />\n";
          data_size += sizeof(uint32_t) + pt_cnt * sizeof(int32_t);
        }
        vtufile << "      </PointData>\n";
        //---------------------------------------------------------------------------
        //---------------------------------------------------------------------------
        vtufile << "      <Cells>\n";
        vtufile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + connect.size() * sizeof(int32_t);
        vtufile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + offset.size() * sizeof(int32_t);
        vtufile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + types.size() * sizeof(uint8_t);
        vtufile << "      </Cells>\n";
        //---------------------------------------------------------------------------
        vtufile << "    </Piece>\n";
        vtufile << "  </UnstructuredGrid>\n";
        // ===========================================================================
        vtufile << "  <AppendedData encoding=\"raw\">\n";
        vtufile << "    _";

        int32_t block_size;
        {  // coord
          block_size = coord.size() * sizeof(VTKReal);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (coord.size()) vtufile.write((char *)&coord[0], coord.size() * sizeof(VTKReal));
        }
        if (value_dof) {  // value
          block_size = value.size() * sizeof(VTKReal);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (value.size()) vtufile.write((char *)&value[0], value.size() * sizeof(VTKReal));
        }
        {  // mpi_rank
          block_size = mpi_rank.size() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (mpi_rank.size()) vtufile.write((char *)&mpi_rank[0], mpi_rank.size() * sizeof(int32_t));
        }
        {  // block_size
          block_size = connect.size() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (connect.size()) vtufile.write((char *)&connect[0], connect.size() * sizeof(int32_t));
        }
        {  // offset
          block_size = offset.size() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (offset.size()) vtufile.write((char *)&offset[0], offset.size() * sizeof(int32_t));
        }
        {  // types
          block_size = types.size() * sizeof(uint8_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (types.size()) vtufile.write((char *)&types[0], types.size() * sizeof(uint8_t));
        }

        vtufile << "\n";
        vtufile << "  </AppendedData>\n";
        // ===========================================================================
        vtufile << "</VTKFile>\n";
      }
      vtufile.close();  // close file
    }
    if (!rank) {  // Write pvtu file
      std::ofstream pvtufile;
      {  // Open file for writing
        std::stringstream pvtufname;
        pvtufname << fname << ".pvtu";
        pvtufile.open(pvtufname.str().c_str());
        if (pvtufile.fail()) return;
      }
      {  // Write to file.
        pvtufile << "<?xml version=\"1.0\"?>\n";
        pvtufile << "<VTKFile type=\"PUnstructuredGrid\">\n";
        pvtufile << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
        pvtufile << "      <PPoints>\n";
        pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\"/>\n";
        pvtufile << "      </PPoints>\n";
        pvtufile << "      <PPointData>\n";
        if (value_dof) {  // value
          pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\"/>\n";
        }
        {  // mpi_rank
          pvtufile << "        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
        }
        pvtufile << "      </PPointData>\n";
        {
          // Extract filename from path.
          std::stringstream vtupath;
          vtupath << '/' << fname;
          std::string pathname = vtupath.str();
          std::string fname_ = pathname.substr(pathname.find_last_of("/\\") + 1);
          // char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
          // std::string fname_ =
          // boost::filesystem::path(fname).filename().string().
          for (int i = 0; i < np; i++) pvtufile << "      <Piece Source=\"" << fname_ << std::setfill('0') << std::setw(6) << i << ".vtu\"/>\n";
        }
        pvtufile << "  </PUnstructuredGrid>\n";
        pvtufile << "</VTKFile>\n";
      }
      pvtufile.close();  // close file
    }
  };
};

#endif
