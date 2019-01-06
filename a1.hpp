#ifndef A1_HPP
#define A1_HPP

#include <vector>
#include <mpi.h>
#include <math.h>

template <typename T, typename Pred>
void mpi_extract_if(MPI_Comm comm, const std::vector<T>& in, std::vector<T>& out, Pred pred) {
    int size, rank; 
    double totalNumOfPred = 0;
    double NumOfPred = 0;

    struct Deficiency {
        double deficiency;
        int rank;
    };

    Deficiency d;
    Deficiency min_d;
    Deficiency max_d;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(auto i: in) {
        if(pred(i)) {
            out.push_back(i);
            NumOfPred += 1;
        }
    }

    MPI_Allreduce(&NumOfPred, &totalNumOfPred, 1, MPI_DOUBLE, MPI_SUM, comm);
    
    double capacity = floor(totalNumOfPred / size);

    d.rank = rank;
    d.deficiency = NumOfPred - capacity;

    MPI_Allreduce(&d, &min_d, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    MPI_Allreduce(&d, &max_d, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

    while(max_d.deficiency > 0 && min_d.deficiency < 0) {

        int destinationRank = min_d.rank;
        int sourceRank = max_d.rank;

        MPI_Datatype MPI_BYTESTREAM;
        MPI_Type_contiguous((-min_d.deficiency)*sizeof(T), MPI_BYTE, &MPI_BYTESTREAM);
        MPI_Type_commit(&MPI_BYTESTREAM);

        if(rank == sourceRank) {
            std::vector<T> elementsToSend(out.end() - std::min<int>(out.size(),(-min_d.deficiency)), out.end());
            out.erase(out.end()- std::min<int>(out.size(),(-min_d.deficiency)), out.end());
            NumOfPred = NumOfPred - (elementsToSend.size());
            MPI_Send(elementsToSend.data(), 1, MPI_BYTESTREAM, destinationRank, 1, comm);
        }
        else if(rank == destinationRank) {
            std::vector<T> elementsReceived(-min_d.deficiency);
            MPI_Recv(elementsReceived.data(), (-min_d.deficiency)*sizeof(T), MPI_BYTE, sourceRank, 1, comm, MPI_STATUS_IGNORE);
            elementsReceived.shrink_to_fit();
            out.insert(out.end(), elementsReceived.begin(), elementsReceived.end());
            NumOfPred = NumOfPred + (elementsReceived.size());
        }

        MPI_Type_free(&MPI_BYTESTREAM);

        d.deficiency = NumOfPred - capacity;
        MPI_Allreduce(&d, &min_d, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
        MPI_Allreduce(&d, &max_d, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    }
} // mpi_extract_if

#endif // A1_HPP
