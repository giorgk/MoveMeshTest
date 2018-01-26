#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H


#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <vector>
#include <math.h>
#include <string>

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


class RBF{
public:
    RBF();

    std::vector<double> centers;
    std::vector<double> width;
    std::vector<double> weights;

   double eval(double x);

   void assign_centers(std::vector<double> cntrs, std::vector<double> wdth);

   void assign_weights(MPI_Comm&  mpi_communicator);

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

void RBF::assign_weights(MPI_Comm&  mpi_communicator){
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
