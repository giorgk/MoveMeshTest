#ifndef MPI_HELP_H
#define MPI_HELP_H

#include <algorithm>

#include <vector>
#include <mpi.h>
#include "pnt_info.h"

/*!
 * \brief Send_receive_size: Each processor sends and receives an integer
 * \param N This is the integer to be sent from this processor
 * \param n_proc The total number of processors
 * \param output A vector of size n_proc which containts the integer that all processors have been sent
 * \param comm The typical MPI communicator
 */
void Send_receive_size(unsigned int N, unsigned int n_proc, std::vector<int> &output, MPI_Comm comm){

        output.clear();
        output.resize(n_proc);
        std::vector<int> temp(n_proc,1);
        std::vector<int> displs(n_proc);
        for (unsigned int i=1; i<n_proc; i++)
                displs[i] = displs[i-1] + 1;

        MPI_Allgatherv(&N, // This is what this processor will send to every other
                       1, //This is the size of the message from this processor
                       MPI_INT, // The data type will be sent
                       &output[0], // This is where the data will be send on each processor
                       &temp[0], // an array with the number of points to be sent/receive
                       &displs[0],
                       MPI_INT, comm);
}

/*!
 * \brief Sent_receive_data: This function sends a vector to all processors and receives all the vectors that the other processor
 * have sent
 * \param data Is a vector of vectors of type T1 with size equal to n_proc.
 * \param N_data_per_proc This is the amount of data that each processor will send.
 * Typically prior to this function the Send_receive_size function should be called to send the sizes
 * \param my_rank the rank of the current processor
 * \param comm The MPI communicator
 * \param MPI_TYPE The mpi type which should match with the templated parameter T1
 */
template <typename T1>
void Sent_receive_data(std::vector<std::vector<T1> > &data,
                       std::vector <int> N_data_per_proc,
                       unsigned int my_rank,
                       MPI_Comm comm,
                       MPI_Datatype MPI_TYPE){

    // data is a vector of vectors of type T1 with size equal to n_proc.
    // This function transfer to all processors the content of data[my_rank]
    // if there are any data in data[i], where i=[1,n_proc; i!=myrank] this will be deleted
    // The size of data[my_rank].size() = N_data_per_proc[my_rank]. This is the responsibility of user

    int N = data[my_rank].size();
    unsigned int n_proc = data.size();
    std::vector<int> displs(n_proc);
    displs[0] = 0;
    for (unsigned int i=1; i<n_proc; i++)
            displs[i] = displs[i-1] + N_data_per_proc[i-1];

    int totdata = displs[n_proc-1] + N_data_per_proc[n_proc-1];
    std::vector<T1> temp_receive(totdata);

    MPI_Allgatherv(&data[my_rank][0], // This is what this processor will send to every other
                   N, //This is the size of the message from this processor
                   MPI_TYPE, // The data type will be sent
                   &temp_receive[0], // This is where the data will be send on each processor
                   &N_data_per_proc[0], // an array with the number of points to be sent/receive
                   &displs[0],
                   MPI_TYPE, comm);

    // Now put the data in the data vector
    for (unsigned int i = 0; i < n_proc; ++i){
            data[i].clear();
            data[i].resize(N_data_per_proc[i]);
            for (int j = 0; j < N_data_per_proc[i]; ++j)
                    data[i][j] = temp_receive[displs[i] +j];
    }
}

/*!
 * \brief This function reads the #i, #j element of a 2D vector after checking
 * whether the indices are in the range of the vector. It is supposed to be a
 * safe way to do v[i][j]
 * \param v the 2D vector.
 * \param i is the index of first element.
 * \param j is the index of the second element.
 */
template <typename T>
T get_v(std::vector<std::vector<T> > v, unsigned int i, unsigned int j){
    if (i < v.size()){
        if (j < v[i].size()){
            return v[i][j];
        }else
            std::cerr << "vector index j:" << j << " out of size: " << v[i].size() << std::endl;
    }else
        std::cerr << "vector index i:" << i << " out of size: " << v.size() << std::endl;
    return 0;
}

/*!
 * \brief Sent_receive_data: This function distributes a vector of type #PntsInfo class.
 * Before the function each processor knows only the values of #Pnts[#my_rank],
 * while after the execution of this function all the  #Pnts[0, ... #my_rank] are populated with values
 * \param Pnts Is a vector of vectors of type PntsInfo<dim> with size equal to n_proc.
 * This is the input and output of this function.
 * As input only the #my_rank element has data, and the ohter elements are emput.
 *  Each point contains, in addition to the standard Point information a list of processors that share
 * the points as well the key where this point can be found.
 * \param my_rank the rank of the current processor
 * \param n_proc is the number of processors
 * \param z_thres is the threshold value in the z direction.
 * \param comm The MPI communicator
 */
template <int dim>
void SendReceive_PntsInfo(std::vector< std::vector<PntsInfo<dim> > > &Pnts,
                          std::map<int , PntsInfo<dim> >& PointsMap,
                          unsigned int my_rank,
                          unsigned int n_proc,
                          double z_thres,
                          MPI_Comm comm){
    std::map<int, int >::iterator it_cn;
    typename std::map<int , PntsInfo<dim> >::iterator it_map;


    std::vector<int> n_dofZ_per_proc(n_proc);
    std::vector<std::vector<int> > dofZ(n_proc);

    // First we will loop through the points that each processor has:
    //Example :
    // -I'm processor 0 and I will send first all the dofs that correspond to my points
    // in addition I'll keep a map where the keys will be the dof and the value a pair of
    // indices
    for (unsigned int i = 0; i < Pnts[my_rank].size(); ++i){
        dofZ[my_rank].push_back(Pnts[my_rank][i].shared_proc.size()); // Number of processors that share the point
        for (unsigned int k = 0; k < Pnts[my_rank][i].shared_proc.size(); ++k){
            dofZ[my_rank].push_back(Pnts[my_rank][i].shared_proc[k]);// the processor that needs this point
            dofZ[my_rank].push_back(Pnts[my_rank][i].key_val_shared_proc[k]);// the key in the processor list that needs this point
        }
        dofZ[my_rank].push_back(Pnts[my_rank][i].Zlist.size()); // the size of the dofs
        for (unsigned int j = 0; j < Pnts[my_rank][i].Zlist.size(); ++j){
            dofZ[my_rank].push_back(Pnts[my_rank][i].Zlist[j].dof);
        }
    }

    //Exchange the dofs of those points
    Send_receive_size(static_cast<unsigned int>(dofZ[my_rank].size()), n_proc, n_dofZ_per_proc, comm);
    Sent_receive_data<int>(dofZ, n_dofZ_per_proc, my_rank, comm, MPI_INT);

    std::vector<std::vector<int> > Request_dof(n_proc);
    // now I will loop through the other processor dofs and I will find out which dofs I dont have and request for the remaining info
    for (unsigned int i = 0; i < n_proc; ++i){
        if (i == my_rank)
            continue;
        std::vector<int> temp_dof_request;
        unsigned int i_cnt = 0;
        for (unsigned j = 0; j < dofZ[i].size(); ++j){
            int Nproc_share = get_v<int>(dofZ, i, i_cnt); i_cnt++;
            //std::vector <int> shared_proc;
            //std::vector <int> shared_proc_keys;
            bool isthisme = false;
            int mykey;
            for (unsigned int k = 0; k < Nproc_share; ++k){
                unsigned int iproc = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                int ikey = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                if (iproc == my_rank){ // if the shared processor is me I should check which dofs I am missing
                    isthisme = true;
                    mykey = ikey;
                }
            }
            if (isthisme)
                it_map = PointsMap.find(mykey);

            int Ndofs = get_v<int>(dofZ, i, i_cnt); i_cnt++;// Number of dofs in this key
            for (unsigned int k = 0; k < Ndofs; ++k){
                int temp_dof = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                if (!isthisme) continue;

                // Check if I need this dof
                if (it_map != PointsMap.end()){
                    // find if the temp dof exist
                    for (unsigned int kk = 0; kk < it_map->second.Zlist.size(); ++kk){
                        if (temp_dof == it_map->second.Zlist[kk].dof){
                            temp_dof_request.push_back(temp_dof);
                            break;
                        }
                    }
                }
            }
        }
        Request_dof[my_rank].push_back(i); // the processor we request the data from
        Request_dof[my_rank].push_back(temp_dof_request.size()); //the total number number of dofs that we want from this processor
        for (unsigned int k = 0; k < temp_dof_request.size(); ++k)
            Request_dof[my_rank].push_back(temp_dof_request[k]);
    }

    std::vector<int> n_request_per_proc(n_proc);
    Send_receive_size(static_cast<unsigned int>(Request_dof[my_rank].size()), n_proc, n_request_per_proc, comm);
    Sent_receive_data<int>(Request_dof, n_request_per_proc, my_rank, comm, MPI_INT);

    {
        // now loop through the requested dofs and pack the data the other processors need
        for (unsigned int i = 0; i < n_proc; ++i){
            if (i == my_rank)
                continue;
            int i_cnt = 0;
            while (i_cnt < Request_dof[i].size()){
                int proc2sent = get_v<int>(Request_dof, i, i_cnt); i_cnt++;
                int Ndofs_from_proc = get_v<int>(Request_dof, i, i_cnt); i_cnt++;
                for (unsigned int j = 0; j < Ndofs_from_proc; ++j){
                    int req_dof = get_v<int>(Request_dof, i, i_cnt); i_cnt++;
                    if (proc2sent == my_rank){// if I am supposed to sent this dof pack its data
                        //@@@@@@@@@@@@@@@@@@@@##############@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    }
                }
            }
            for (int j = 0; j < Request_dof[i].size(); ++j){


            }

        }

    }




    // serialize the information into 2 vectors int and double
    std::vector<int> n_int_per_proc;
    std::vector<int> n_dbl_per_proc;
    std::vector<std::vector<int> > int_data(n_proc);
    std::vector<std::vector<double> > dbl_data(n_proc);
    int_data[my_rank].push_back(Pnts[my_rank].size()); // The 1st int is the number of Pnts NPnt this processor will transfer
    for (unsigned int i = 0; i < Pnts[my_rank].size(); ++i){
        //if (my_rank == 0) std::cout << "x: " << Pnts[my_rank][i].PNT[0] << std::endl;
        dbl_data[my_rank].push_back(Pnts[my_rank][i].PNT[0]);
        if (dim == 3){
            //if (my_rank == 0) std::cout << "y " << Pnts[my_rank][i].PNT[1] << std::endl;
            dbl_data[my_rank].push_back(Pnts[my_rank][i].PNT[1]);
        }

        // For each pnt set the number of processors that share this point
        int_data[my_rank].push_back(Pnts[my_rank][i].shared_proc.size());
        // Next send the processors that should take this point
        for (unsigned int ii = 0; ii < Pnts[my_rank][i].shared_proc.size(); ++ii){
            int_data[my_rank].push_back(Pnts[my_rank][i].shared_proc[ii]);
        }
        // Then send the number of points Nz in the Zlist list
        int_data[my_rank].push_back(Pnts[my_rank][i].Zlist.size()); //number of Z coordinates Nz in the list
        for (unsigned int j = 0; j < Pnts[my_rank][i].Zlist.size(); ++j){ // then send the following info Nz times
            int_data[my_rank].push_back(Pnts[my_rank][i].Zlist[j].dof);     // dof
            if (Pnts[my_rank][i].Zlist[j].isTop == 1) // instead of sending both top and bottom we will send 1 int depending the case
                int_data[my_rank].push_back(1);
            else if (Pnts[my_rank][i].Zlist[j].isBot == 1)
                int_data[my_rank].push_back(2);
            else
                int_data[my_rank].push_back(0);
            int_data[my_rank].push_back(Pnts[my_rank][i].Zlist[j].dof_conn.size());   // Send the number of connections Nconn for this point
            for (it_cn = Pnts[my_rank][i].Zlist[j].dof_conn.begin(); it_cn != Pnts[my_rank][i].Zlist[j].dof_conn.end(); ++it_cn){// for each Nconn send the following
                int_data[my_rank].push_back(it_cn->first); // the dof of the connected node
                int_data[my_rank].push_back(it_cn->second); // hanging of the connected node
            }
            int_data[my_rank].push_back(Pnts[my_rank][i].Zlist[j].cnstr_nds.size());   // Send the number of constraints Nconn for this point
            for (int jj = 0; jj < Pnts[my_rank][i].Zlist[j].cnstr_nds.size(); ++jj){// send the dofs that constraint this node
                int_data[my_rank].push_back(Pnts[my_rank][i].Zlist[j].cnstr_nds[jj]);
            }
            // And last add the z value in the double vector
            //if (my_rank == 0) std::cout << "z: " << Pnts[my_rank][i].Zlist[j].z << std::endl;
            dbl_data[my_rank].push_back(Pnts[my_rank][i].Zlist[j].z);
        }
    }
    std::cout << "I'm rank " << my_rank << " and have to send " << dbl_data[my_rank].size()
              << " doubles and " << int_data[my_rank].size() << " ints from " << int_data[my_rank][0] << " points" << std::endl;


    // ------------------------- PASS DATA TO OTHER PROCESSORS
    Send_receive_size(static_cast<unsigned int>(int_data[my_rank].size()), n_proc, n_int_per_proc, comm); // number of integers to transfer
    Send_receive_size(static_cast<unsigned int>(dbl_data[my_rank].size()), n_proc, n_dbl_per_proc, comm); // number of doubles to transfer

    Sent_receive_data<int>(int_data, n_int_per_proc, my_rank, comm, MPI_INT);
    Sent_receive_data<double>(dbl_data, n_dbl_per_proc, my_rank, comm, MPI_DOUBLE);

    //for (unsigned int i = 0; i < n_proc; ++i){
    //    std::cout << "------I'm rank " << my_rank << " and I have from (" << i << ") " << int_data[i].size()
    //              << " ints, and " <<  dbl_data[i].size() << " doubles------" << std::endl;
    //}

    // -----------------------------------------------------------------------------

    // loop through the data and complete the Pnts vector for each processor
    //int dbg_proc = 2;

    for (unsigned int i_proc = 0; i_proc < int_data.size(); ++i_proc){ // loop through the processors
        if (i_proc == my_rank)
            continue;
        unsigned int i_cnt = 0;
        unsigned int d_cnt = 0;

        // How many points my_rank gets from processor i_proc:
        int Npnts = get_v<int>(int_data, i_proc, i_cnt);  i_cnt++;
        //if (my_rank == dbg_proc) std::cout << "=========I'm rank " << my_rank << " and proc " << i_proc << " sends me " << Npnts << " pnts" << std::endl;


        for (int i = 0; i < Npnts; ++i){
            double x, y, z;
            x = get_v<double>(dbl_data, i_proc, d_cnt);d_cnt++;
            //if (my_rank == dbg_proc) std::cout << "-------I'm rank " << my_rank << ", x: " << x << std::endl;
            if (dim == 3){
                y = get_v<double>(dbl_data, i_proc, d_cnt);d_cnt++;
            }
            int Nshared_proc = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
            std::vector<int> shared_proc;
            for (int ii = 0; ii < Nshared_proc; ++ii){
                shared_proc.push_back(get_v<int>(int_data, i_proc, i_cnt)); i_cnt++;
            }

            int Nz = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
            //if (my_rank == dbg_proc) std::cout << "......I'm rank " << my_rank << ", Nz: " << Nz << std::endl;

            for (int j = 0; j < Nz; ++j){
                z = get_v<double>(dbl_data, i_proc, d_cnt); d_cnt++;
                //if (my_rank == dbg_proc) std::cout << "- - - -I'm rank " << my_rank << ", z: " << z << std::endl;
                int dof = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                //if (my_rank == dbg_proc) std::cout << "I'm rank " << my_rank << ", dof: " << dof << std::endl;
                int top_bot_flag = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                int istop, isbot;
                if (top_bot_flag == 1) {istop = 1;isbot=0;}
                else if (top_bot_flag == 2){istop = 0;isbot=1;}
                else{istop = 0;isbot=0;}

                int Nconn = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                //if (my_rank == dbg_proc) std::cout << "I'm rank " << my_rank << ", Nconn: " << Nconn << std::endl;
                std::map<int, int> conn;
                for (int k = 0; k < Nconn; ++k){
                    int c_dof = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    //if (my_rank == dbg_proc) std::cout << "I'm rank " << my_rank << ", c_dof: " << c_dof << std::endl;
                    int c_hng = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    //if (my_rank == dbg_proc) std::cout << "I'm rank " << my_rank << ", c_hng: " << c_hng << std::endl;
                    conn.insert(std::pair<int, int>(c_dof, c_hng));
                }

                int Ncnsrt = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                std::vector<int> cnstr_dofs;
                for (int k = 0; k < Ncnsrt; ++k){
                    cnstr_dofs.push_back(get_v<int>(int_data, i_proc, i_cnt));i_cnt++;
                }

                Zinfo zinfo(z, dof, cnstr_dofs, istop, isbot, conn);
                Point<dim-1> p;
                p[0] = x;
                if (dim == 3)
                    p[1] = y;

                if (j == 0){
                    PntsInfo<dim> pntinfo(p,zinfo);
                    pntinfo.shared_proc = shared_proc;
                    Pnts[i_proc].push_back(pntinfo);
                }else{
                    Pnts[i_proc][Pnts[i_proc].size()-1].add_Zcoord(zinfo, z_thres);
                }
            }
        }
    }
}




#endif // MPI_HELP_H
