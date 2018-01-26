#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H


#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <vector>
#include <math.h>
#include <string>
#include "mpi_help.h"

using namespace dealii;


/*!
 * \brief linspace generate a linearly spaced vector between the two numbers min and max
 * \param min is the lower end
 * \param max is the upper end
 * \param n is the number of numbers to generate between the min and max
 * \return a vector of n numbers linearly spaced
 */
std::vector<double> linspace(double min, double max, int n){
    std::vector<double> result;
    int iterator = 0;
    for (int i = 0; i <= n-2; i++){
        double temp = min + i*(max-min)/(floor(static_cast<double>(n)) - 1);
        result.insert(result.begin() + iterator, temp);
        iterator += 1;
    }
    result.insert(result.begin() + iterator, max);
    return result;
}

/*!
 * \brief is_input_a_scalar check if the string can be converted into a scalar value
 * \param input is the string to test
 * \return true if the input can be a scalars
 */
bool is_input_a_scalar(std::string input){
    // try to convert the input to scalar
    bool outcome;
    try{
        double value = stod(input);
        value++; // something to surpress the warning
        outcome = true;
    }
    catch(...){
        outcome = false;
    }
    return outcome;
}

/*!
 * \brief This function returns a list of indices that are connected to the #ii node in a cell
 * \param #ii is the index of the node we seek its connected nodes
 * \return a vector of connected indices. Not that this is simply returns the ids of the nodes
 * as they are defined by the geometry info class
 */
template <int dim>
std::vector<int> get_connected_indices(int ii){
    std::vector<int> out;
    if (dim == 2){
        if (ii == 0){
            out.push_back(1);
            out.push_back(2);
        }else if (ii == 1){
            out.push_back(0);
            out.push_back(3);
        }else if (ii == 2){
            out.push_back(0);
            out.push_back(3);
        }else if (ii == 3){
            out.push_back(1);
            out.push_back(2);
        }
    }else if (dim == 3){
        if (ii == 0){
            out.push_back(1);
            out.push_back(2);
            out.push_back(4);
        }else if (ii == 1){
            out.push_back(0);
            out.push_back(3);
            out.push_back(5);
        }else if (ii == 2){
            out.push_back(0);
            out.push_back(3);
            out.push_back(6);
        }else if (ii == 3){
            out.push_back(1);
            out.push_back(2);
            out.push_back(7);
        }else if (ii == 4){
            out.push_back(0);
            out.push_back(5);
            out.push_back(6);
        }else if (ii == 5){
            out.push_back(1);
            out.push_back(4);
            out.push_back(7);
        }else if (ii == 6){
            out.push_back(2);
            out.push_back(4);
            out.push_back(7);
        }else if (ii == 7){
            out.push_back(3);
            out.push_back(5);
            out.push_back(6);
        }
    }
    return out;
}

template <int dim>
void create_outline_polygon(std::vector<std::vector<Point<dim-1>>> &pointdata, MPI_Comm  mpi_communicator){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
    if (dim == 2){
        Point<dim-1> minX; minX[0] = 1000000000;
        Point<dim-1> maxX; maxX[0] = -1000000000;
        for (unsigned int i = 0; i <pointdata[my_rank].size(); ++i){
            if (pointdata[my_rank][i][0] < minX[0])
                minX[0] = pointdata[my_rank][i][0];
            if (pointdata[my_rank][i][0] > maxX[0])
                maxX[0] = pointdata[my_rank][i][0];
        }
        pointdata[my_rank].clear();
        pointdata[my_rank].push_back(minX);
        pointdata[my_rank].push_back(maxX);
        std::cout << "I'm rank " << my_rank << " and my limits are: (" << pointdata[my_rank][0][0] << ", " << pointdata[my_rank][1][0] << ")" << std::endl;

    }else if (dim == 3){
        std::cerr << "Not implemented yet" << std::endl;
    }else{
        std::cerr << "Unsuported dimension. Has to be 2D or 3D" << std::endl;
    }

    // Send my polygon outline to all processors
    std::vector<std::vector<double> > serialized_points(n_proc);
    for (unsigned int i = 0; i < pointdata[my_rank].size(); ++ i){
        serialized_points[my_rank].push_back(pointdata[my_rank][i][0]);
        if (dim == 3)
            serialized_points[my_rank].push_back(pointdata[my_rank][i][1]);
    }

    //std::cout << "I'm " << my_rank << " and have " << serialized_points[my_rank].size() << " serialized points" << std::endl;

    std::vector <int> Npoints_per_polygon_proc(n_proc);
    Send_receive_size(serialized_points[my_rank].size(), n_proc, Npoints_per_polygon_proc, mpi_communicator);
    //for (int i = 0; i < n_proc; ++i)
    //    std::cout << "I'm " << my_rank << " and proc " << i << " has " << Npoints_per_polygon_proc[i] << " points" << std::endl;


    Sent_receive_data<double>(serialized_points, Npoints_per_polygon_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    //if (my_rank == 3){
    //    for (int i = 0; i < n_proc; ++i){
    //        std::cout << i << " : " << serialized_points[i][0] << ", " << serialized_points[i][1] << std::endl;
    //    }
    //}

    // gather data from the other processors
    for (unsigned int i = 0; i < n_proc; ++i){
        if (i == my_rank)
            continue;
        for (unsigned int j = 0; j < Npoints_per_polygon_proc[my_rank];){
            Point<dim-1> tempP;
            tempP[0] = serialized_points[i][j]; j++;
            if (dim == 3){
                tempP[1] = serialized_points[i][j]; j++;
            }
            pointdata[i].push_back(tempP);
        }
    }
}

template <int dim>
std::vector<int> send_point(Point<dim-1> p, std::vector<std::vector<Point<dim-1>>> pointdata, int my_rank){
    std::vector<int> shared_proc;
    if (dim == 2){
        for (unsigned int i = 0; i < pointdata.size(); ++i){
            if (i == my_rank)
                continue;
            double xmin = pointdata[i][0][0];
            double xmax = pointdata[i][1][0];
            if (p[0] > xmin-0.01 && p[0] < xmax + 0.01){
                shared_proc.push_back(i);
            }
        }
    } else if (dim == 3) {
        std::cerr << "3D not implemented yet" << std::endl;
    }
    return shared_proc;
}


class RBF{
public:
    RBF();

    std::vector<double> centers;
    std::vector<double> width;
    std::vector<double> weights;

   double eval(double x);

   void assign_centers(std::vector<double> cntrs, std::vector<double> wdth);

   void assign_weights(MPI_Comm  mpi_communicator);

};

RBF::RBF(){}

double RBF::eval(double x){
    if (centers.size() != weights.size())
        std::cerr << "The weights have to be equal with the centers" << std::endl;

    double v = 0;
    for (unsigned int i = 0; i < centers.size(); ++i){
        v += weights[i]*exp(-pow(width[i]*fabs(x - centers[i]),2));
    }
    return v;
}

void RBF::assign_centers(std::vector<double> cntrs, std::vector<double> wdth){
    for (unsigned int i = 0; i < cntrs.size(); ++i){
        centers.push_back(cntrs[i]);
        width.push_back(wdth[i]);
    }
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void RBF::assign_weights(MPI_Comm  mpi_communicator){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    MPI_Barrier(mpi_communicator);
    if (my_rank == 0){
        weights.clear();
        for (unsigned int i = 0; i < centers.size(); ++i){
            weights.push_back(fRand(-100, 100));
        }
    }else{
        weights.clear();
        weights.resize(centers.size());
    }
    MPI_Barrier(mpi_communicator);
    MPI_Bcast(&weights[0], static_cast<int>(centers.size()),MPI_DOUBLE,0,mpi_communicator);
    MPI_Barrier(mpi_communicator);
    //std::cout << "I'm rank " << my_rank << " with " << weights.size() << " weights" << std::endl;
    //for (int i = 0; i < weights.size(); ++i){
    //    std::cout << "rank" << my_rank << ", w" << i << "=" << weights[i] << std::endl;
    //}
}




#endif // HELPER_FUNCTIONS_H
