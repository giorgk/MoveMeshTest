#ifndef MESH_STRUCT_H
#define MESH_STRUCT_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>

#include "zinfo.h"
#include "pnt_info.h"
#include "cgal_functions.h"
#include "my_functions.h"
#include "mpi_help.h"
#include "helper_functions.h"

//! custom struct to hold data
template <int dim>
struct trianode {
    Point <dim> pnt;
    int dof;
    int level;
    int hang;
    int spi; // support_point_index
    std::map<int,int> c_pnt;// dofs of points connected to that node
};


struct PntIndices{
    int XYind;
    int Zind;
};


//! Returns true if any neighbor element is ghost
template <int dim>
bool any_ghost_neighbor(typename DoFHandler<dim>::active_cell_iterator cell){
    bool out = false;
    for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface){
        if (cell->at_boundary(iface)) continue;

        if (cell->neighbor(iface)->active()){
            if (cell->neighbor(iface)->is_ghost()){
                out = true;
                break;
            }
        }
        else{
            for (unsigned int ichild = 0; ichild < cell->face(iface)->n_children();  ++ichild){
                if (cell->neighbor_child_on_subface(iface,ichild)->is_ghost()){
                    out = true;
                    break;
                }
            }
            if (out)
                break;
        }
    }
    return out;
}

/*!
 * \brief The Mesh_struct class contains the coordinates of the entire mesh grouped into lists of dim-1 points,
 * where each point contains a list of the z node elevation.
 */
template <int dim>
class Mesh_struct{
public:

    /*!
     * \brief Mesh_struct The constructor just sets the threshold values
     * \param xy_thr this is the threshold for the x-y coordinates. Two point with distance smaller that xy_thr
     * are considered identical. Typically the xy threshold has larger values compared to the z threshold.
     * \param z_thr this is the treshold for the z coordinates. Two nodes with the same x-y coordinates
     * are considered identical if their elevation is smaller that the z_thr
     */
    Mesh_struct(double xy_thr, double z_thr);

    //! The threshold along the x-y coordinates
    double xy_thres;
    //! The threshold along the z coordinates
    double z_thres;

    //! This is a counter for the points in the #PointsMap
    int _counter;

    //! This map associates each point with a unique id (#_counter)
    std::map<int , PntsInfo<dim> > PointsMap;

    //! This is a Map structure that relates the dofs with the PointsMap.
    //! The key is the dof and the value is the pair #PointsMap key and the index of the z value in
    //! the Zlist of the #PointsMap.
    //! In other words <dof> - <xy_index, z_index>
    std::map<int,std::pair<int,int> > dof_ij;

    //! this is a cgal container of the points of this class stored in an optimized way for spatial queries
    PointSet2 CGALset;

    //! THe number of levels in the mesh starting from 0 for the coarsest nodes
    int n_levels;

    //! Adds a new point in the structure. If the point exists adds the z coordinate only and returns
    //! the id of the existing point. if the point doesnt exist creates a new point and returns the new id.
    PntIndices add_new_point(Point<dim-1>, Zinfo zinfo);

    //! Checks if the point already exists in the mesh structure
    //! If the point exists it returns the id of the point in the #CGALset
    //! otherwise returns -9;
    int check_if_point_exists(Point<dim-1> p);

    /*!
     * \brief updateMeshstruct is the heart of this class. for a given parallel triangulation updates the existing
     * points or creates new ones.
     *
     * The method first loops through the locally owned cells and extracts the coordinates and dof for each coordinate
     * which stores it to #distributed_mesh_vertices.
     * \param xy_thr this is the threshold for the x-y coordinates. Two point with distance smaller that xy_thr
     * are considered identical. Typically the xy threshold has larger values compared to the z threshold.
     * \param distributed_mesh_vertices is a vector of size #dim x (Number of triangulation vertices).
     * Essentially we treat all the coordinates as unknowns yet only the vertical component is the one we are going
     * to change
     */
    void updateMeshStruct(DoFHandler<dim>& mesh_dof_handler,
                         FESystem<dim>& mesh_fe,
                         ConstraintMatrix& mesh_constraints,
                         IndexSet& mesh_locally_owned,
                         IndexSet& mesh_locally_relevant,
                         TrilinosWrappers::MPI::Vector& mesh_vertices,
                         TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                         MPI_Comm&  mpi_communicator,
                         ConditionalOStream pcout);

    //! Once the #PointsMap::T and #PointsMap::B have been set to a new elevation and have also
    //! the relative positions calculated we can use this method to update the z elevations of the
    //! Mesh structure. Then the updated elevations will be transfered to the mesh.
    //! The update starts with the nodes at level 0, which can be set directly as they do not depend on
    //! any other node.
    void updateMeshElevation(DoFHandler<dim>& mesh_dof_handler,
                             ConstraintMatrix& mesh_constraints,
                             TrilinosWrappers::MPI::Vector& mesh_vertices,
                             TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                             MPI_Comm&  mpi_communicator,
                             ConditionalOStream pcout);

    //! resets all the information that is contained except the coordinates and the level of the points
    void reset();

    //! Prints to screen the number of vertices the #myrank processor has.
    //! It is used primarily for debuging
    void n_vertices(int myrank);

    //! Assuming that all processors have gathered all Z nodes for each xy point they own
    //! this routine identifies the dofs above and below each node, how the nodes are connected,
    //! and sets the top and bottom elevation for each xy point
    //! (MAYBE THIS SHOULD SET THE DOF of the top/bottom node and not the elevation
    void set_id_above_below();

    //! This creates the #dof_ij map.
    void make_dof_ij_map();

    //! This method calculates the top and bottom elevation on the points of the #PointsMap
    //! This should be called on the initial grid before any refinement
    void compute_initial_elevations(MyFunction<dim, dim-1> top_function,
                                    MyFunction<dim, dim-1> bot_function,
                                    std::vector<double>& vert_discr);

    //! Calculates the positions of the vertices that belong to a given level #level
    void update_z(int level, MPI_Comm &mpi_communicator);

    //! This method sets the scales #dbg_scale_x and #dbg_scale_z for debug plotting using softwares like houdini
    void dbg_set_scales(double xscale, double zscale);

    //! Update level
    void update_level(int new_level);

private:
    void dbg_meshStructInfo2D(std::string filename, unsigned int n_proc);
    void dbg_meshStructInfo3D(std::string filename, unsigned int n_proc);
    double dbg_scale_x;
    double dbg_scale_z;


};

template <int dim>
Mesh_struct<dim>::Mesh_struct(double xy_thr, double z_thr){
    xy_thres = xy_thr;
    z_thres = z_thr;
    _counter = 0;
    dbg_scale_x = 100;
    dbg_scale_z = 100;
    n_levels = 0;
}

template <int dim>
PntIndices Mesh_struct<dim>::add_new_point(Point<dim-1>p, Zinfo zinfo){
    PntIndices outcome;
    outcome.XYind = -99;
    outcome.Zind = -99;

    // First search for the XY location in the structure
    int id = check_if_point_exists(p);
    if ( id < 0 ){
        // this is a new point and we add it to the map
        PntsInfo<dim> tempPnt(p, zinfo);
        PointsMap[_counter] = tempPnt;

        //... to the Cgal structure
        std::vector< std::pair<ine_Point2,unsigned> > pair_point_id;
        double x,y;
        if (dim == 2){
            x = p[0];
            y = 0;
        }else if (dim == 3){
            x = p[0];
            y = p[1];
        }
        pair_point_id.push_back(std::make_pair(ine_Point2(x, y), _counter));
        CGALset.insert(pair_point_id.begin(), pair_point_id.end());
        outcome.XYind = _counter;
        _counter++;
    }else if (id >=0){
        typename std::map<int, PntsInfo<dim> >::iterator it = PointsMap.find(id);
        it->second.add_Zcoord(zinfo, z_thres);
        outcome.XYind = it->first;
    }

    return outcome;

}

template <int dim>
int Mesh_struct<dim>::check_if_point_exists(Point<dim-1> p){
    int out = -9;
    double x,y;
    if (dim == 2){
        x = p[0];
        y = 0;
    }else if (dim == 3){
        x = p[0];
        y = p[1];
    }

    std::vector<int> ids = circle_search_in_2DSet(CGALset, ine_Point3(x, y, 0.0) , xy_thres);

    if (ids.size() > 1)
        std::cerr << "More than one points around x: " << x << ", y: " << y << "found within the specified tolerance" << std::endl;
    else if(ids.size() == 1) {
         out = ids[0];
    }
    else{
         out = -9;
    }

    return out;
}

template <int dim>
void Mesh_struct<dim>::update_level(int new_level){
    if (new_level > n_levels)
        n_levels = new_level;
}

template <int dim>
void Mesh_struct<dim>::updateMeshStruct(DoFHandler<dim>& mesh_dof_handler,
                                       FESystem<dim>& mesh_fe,
                                       ConstraintMatrix& mesh_constraints,
                                       IndexSet& mesh_locally_owned,
                                       IndexSet& mesh_locally_relevant,
                                       TrilinosWrappers::MPI::Vector& mesh_vertices,
                                       TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                                       MPI_Comm&  mpi_communicator,
                                       ConditionalOStream pcout){
    // Use this to time the operation. Note that this is a very expensive operation but nessecary
    std::clock_t begin_t = std::clock();
    // get the rank and processor id just for output display
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    // make sure all processors start together
    MPI_Barrier(mpi_communicator);
    reset(); // delete all the information except the coordinates
    MPI_Barrier(mpi_communicator);

    const MappingQ1<dim> mapping;

    pcout << "Distribute mesh dofs..." << std::endl << std::flush;
    mesh_dof_handler.distribute_dofs(mesh_fe);
    mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);
    mesh_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    distributed_mesh_vertices.reinit(mesh_locally_owned, mpi_communicator);

    const std::vector<Point<dim> > mesh_support_points
                                  = mesh_fe.base_element(0).get_unit_support_points();

    FEValues<dim> fe_mesh_points (mapping,
                                  mesh_fe,
                                  mesh_support_points,
                                  update_quadrature_points);

    mesh_constraints.clear();
    mesh_constraints.reinit(mesh_locally_relevant);
    DoFTools::make_hanging_node_constraints(mesh_dof_handler, mesh_constraints);
    mesh_constraints.close();

    // to avoid duplicate executions we will maintain a map with the dofs that have been
    // already processed
    std::map<int,int> dof_local;// the key are the dof and the value is the _counter
    std::map<int,int>::iterator itint;
    MPI_Barrier(mpi_communicator);

    pcout << "Update XYZ structure..." << std::endl << std::flush;
    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell = mesh_dof_handler.begin_active(),
        endc = mesh_dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            fe_mesh_points.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);
            // First we will loop through the cell dofs gathering all info we need for the points
            // and then we will loop again though the points to add the into the structure.
            // Therefore we would need to initialize several vectors
            std::map<int, trianode<dim> > curr_cell_info;


            for (unsigned int idof = 0; idof < mesh_fe.base_element(0).dofs_per_cell; ++idof){
                // for each dof of this cell we extract the coordinates and the dofs
                Point <dim> current_node;
                std::vector<int> current_dofs(dim);
                std::vector<unsigned int> spi;
                for (unsigned int dir = 0; dir < dim; ++dir){
                    // for each cell support_point_index spans from 0 to dim*Nvert_per_cell-1
                    // eg for dim =2 spans from 0-7
                    // The first dim indices correspond to x,y,z of the first vertex of triangulation
                    // The current_dofs contains the dof index for each coordinate.
                    // The current_node containts the x,y,z coordinates
                    // The distributed_mesh_vertices is a vector of size Nvertices*dim
                    // essentially we are treating all xyz coordinates as variables although we are going to
                    // change only the vertical component of it (In 2D this is the y).
                    unsigned int support_point_index = mesh_fe.component_to_system_index(dir, idof );
                    spi.push_back(support_point_index);
                    current_dofs[dir] = static_cast<int>(cell_dof_indices[support_point_index]);
                    current_node[dir] = fe_mesh_points.quadrature_point(idof)[dir];
                    distributed_mesh_vertices[cell_dof_indices[support_point_index]] = current_node[dir];

                    //pcout << "dir:" << dir << ", idof:" << idof << ", cur_dof:" << current_dofs[dir]
                    //      <<   ", cur_nd:" << current_node[dir] << ", spi:" << support_point_index << std::endl;
                }
                trianode<dim> temp;
                temp.pnt = current_node;
                temp.dof = current_dofs[dim-1];
                temp.level = cell->level();
                temp.hang = mesh_constraints.is_constrained(current_dofs[dim-1]);
                temp.spi = spi[dim-1];
                curr_cell_info[idof] = temp;
            }

            typename std::map<int, trianode<dim> >::iterator it;
            for (it = curr_cell_info.begin(); it != curr_cell_info.end(); ++it){
                // get the nodes connected with this one
                std::vector<int> id_conn = get_connected_indices<dim>(it->first);
                // create a map of the points to add
                std::map<int, std::pair<int,int> > connectedNodes;
                for (unsigned int i = 0; i < id_conn.size(); ++i){
                    connectedNodes.insert(std::pair<int, std::pair<int,int> >(curr_cell_info[id_conn[i]].dof,
                                          std::pair<int,int> (curr_cell_info[id_conn[i]].level,
                                                              curr_cell_info[id_conn[i]].hang)));
                }

                // Now create a zinfo variable
                Zinfo zinfo(it->second.pnt[dim-1], it->second.dof, it->second.level,it->second.hang, connectedNodes);
                update_level(it->second.level);
                // and a point
                Point<dim-1> ptemp;
                for (unsigned int d = 0; d < dim-1; ++d)
                    ptemp[d] = it->second.pnt[d];

                // Try to add it in the structure
                PntIndices id_in_map = add_new_point(ptemp, zinfo);

                if (id_in_map.XYind < 0)
                    std::cerr << "Something went really wrong while trying to insert a new point into the mesh struct" << std::endl;
                else{
                    bool tf = any_ghost_neighbor<dim>(cell);
                    if (tf)
                        PointsMap[id_in_map.XYind].have_to_send = 1;
                }
            }
            //pcout << "----------------------------------------------------------------" << std::endl;
        }
    }

    make_dof_ij_map();



    MPI_Barrier(mpi_communicator);
    distributed_mesh_vertices.compress(VectorOperation::insert);
    MPI_Barrier(mpi_communicator);

    //dbg_meshStructInfo2D("before2D", my_rank);
    dbg_meshStructInfo3D("before3D", my_rank);


    if (n_proc > 1){
        pcout << "exchange vertices between processors..." << std::endl << std::flush;
        // All vertices have been added to the PointsMap structure.
        // we loop through each vertex and store to a separate vector
        // those that require communication and they are actively used

        std::vector< std::vector<PntsInfo<dim> > > sharedPoints(n_proc);

        typename std::map<int ,  PntsInfo<dim> >::iterator it;
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            if (it->second.have_to_send == 1){
                // IN the old code I was checking for positive dofs.
                // I have to see whether I should check that again
                sharedPoints[my_rank].push_back(it->second);
            }
        }

        //std::cout << "I'm rank " << my_rank << " and I'll send " << sharedPoints[my_rank].size() << std::endl;
        MPI_Barrier(mpi_communicator);

        // -----------------Send those points to every processor------------

        SendReceive_PntsInfo(sharedPoints, my_rank, n_proc, z_thres, mpi_communicator);

        // Loop through the received points and get the ones my_rank needs
        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (i_proc == my_rank) continue;// my_rank already knows these poitns

            for (unsigned int i = 0; i < sharedPoints[i_proc].size(); ++i){
                int id = check_if_point_exists(sharedPoints[i_proc][i].PNT);
                if (id >= 0){
                    it = PointsMap.find(id);
                    if (it == PointsMap.end())
                        std::cerr << "There must be an entry under this key" << std::endl;
                    else{
                        std::vector<Zinfo>::iterator itz = sharedPoints[i_proc][i].Zlist.begin();
                        for (; itz != sharedPoints[i_proc][i].Zlist.end(); ++itz){
                            it->second.add_Zcoord(*itz, z_thres);
                        }
                    }
                }
            }
        }
        MPI_Barrier(mpi_communicator);
    }//if (n_proc > 1)

    // IN the new code this may not needed. However it is needed for debuging
    make_dof_ij_map();
    //dbg_meshStructInfo3D("After3D", my_rank);

    set_id_above_below();
    //dbg_meshStructInfo3D("After3D", my_rank);

    std::clock_t end_t = std::clock();
    double elapsed_secs = double(end_t - begin_t)/CLOCKS_PER_SEC;
    //std::cout << "====================================================" << std::endl;
    std::cout << "I'm rank " << my_rank << " and spend " << elapsed_secs << " sec on Updating XYZ" << std::endl;
    //std::cout << "====================================================" << std::endl;
}

template <int dim>
void Mesh_struct<dim>::reset(){
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        it->second.reset();
    }
    dof_ij.clear();
}

template <int dim>
void Mesh_struct<dim>::n_vertices(int myrank){
    int Nxy = PointsMap.size();
    int Nz = 0;
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        Nz += it->second.Zlist.size();
    }
    std::cout << "I'm " << myrank << ", Nxy = " << Nxy << ", Nz = " << Nz << std::endl;
}

template <int dim>
void Mesh_struct<dim>::dbg_meshStructInfo2D(std::string filename, unsigned int my_rank){
    const std::string log_file_name = (filename	+ "_" +
                                       Utilities::int_to_string(my_rank+1, 4) +
                                       ".txt");
     std::ofstream log_file;
     log_file.open(log_file_name.c_str());
     typename std::map<int , PntsInfo<dim> >::iterator it;
     for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
         double x,y,z;
         x = it->second.PNT[0]/dbg_scale_x;
         if (dim == 3) z = it->second.PNT[1]/dbg_scale_x;
         else z = 0;
         y = 0;
         log_file << std::setprecision(3)
                  << std::fixed
                  << std::setw(15) << x << ", "
                  << std::setw(15) << y << ", "
                  << std::setw(15) << z << ", "
                  << std::setw(15) << it->second.T << ", "
                  << std::setw(15) << it->second.B << ", "
                  << std::setw(5) << it->second.have_to_send
                  << std::endl;
     }
     log_file.close();

}

template <int dim>
void Mesh_struct<dim>::dbg_meshStructInfo3D(std::string filename, unsigned int my_rank){
    const std::string log_file_name = (filename + "_pnt_" +
                                       Utilities::int_to_string(my_rank+1, 4) +
                                       ".txt");
     std::ofstream log_file;
     log_file.open(log_file_name.c_str());

     std::map<std::pair<int,int>, int> line_map;
     std::pair<std::map<std::pair<int,int>,int>::iterator,bool> ret;
     int counter = 0;

     typename std::map<int , PntsInfo<dim> >::iterator it;
     for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
         std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
         for (; itz != it->second.Zlist.end(); ++itz){
             double x,y,z;
             x = it->second.PNT[0]/dbg_scale_x;
             if (dim == 3) z = it->second.PNT[1]/dbg_scale_x;
             else z = 0;
             y = itz->z/dbg_scale_z;
             log_file << std::setprecision(3)
                      << std::fixed
                      << std::setw(15) << x << ", "
                      << std::setw(15) << y << ", "
                      << std::setw(15) << z << ", "
                      << std::setw(15) << itz->dof << ", "
                      << std::setw(15) << itz->dof_conn.size() << ", "
                      << std::setw(15) << itz->level << ", "
                      << std::setw(15) << itz->id_above  << ", "
                      << std::setw(15) << itz->id_below << ", "
                      << std::setw(15) << itz->id_top  << ", "
                      << std::setw(15) << itz->id_bot << ", "
                      << std::setw(15) << itz->rel_pos  << ", "
                      << std::setw(15) << itz->hanging << ", "
                      << std::setw(15) << it->second.T << ", "
                      << std::setw(15) << it->second.B << ", "
                      << std::setw(5) << it->second.have_to_send
                      << std::endl;


             std::map<int,std::pair<int,int> >::iterator itt;
             for (itt = itz->dof_conn.begin(); itt != itz->dof_conn.end(); ++itt){
                 int a,b;
                 if (itz->dof < itt->first){
                     a = itz->dof;
                     b = itt->first;
                 }
                 else{
                     a = itt->first;
                     b = itz->dof;
                 }
                 ret = line_map.insert(std::pair<std::pair<int,int>,int>(std::pair<int,int>(a,b),counter));
                 if (ret.second == true)
                    counter++;
             }
         }
     }
     log_file.close();

     // Print the lines hopefully these are unique

     const std::string log_file_name1 = (filename + "_lns_" +
                                        Utilities::int_to_string(my_rank+1, 4) +
                                        ".txt");

     std::ofstream log_file1;
     log_file1.open(log_file_name1.c_str());

     std::map<int,std::pair<int,int> >::iterator it_dof;
     std::map<std::pair<int,int>, int>::iterator itl;
     double x1,y1,z1,x2,y2,z2;
     for (itl = line_map.begin(); itl!=line_map.end(); ++itl){
        it_dof = dof_ij.find(itl->first.first);
        if (it_dof != dof_ij.end()){
            x1 = PointsMap[it_dof->second.first].PNT[0];
            if (dim ==2 ) z1 = 0; else
                z1 = PointsMap[it_dof->second.first].PNT[1];
            y1 = PointsMap[it_dof->second.first].Zlist[it_dof->second.second].z;

            it_dof = dof_ij.find(itl->first.second);
            if (it_dof != dof_ij.end()){
                x2 = PointsMap[it_dof->second.first].PNT[0];
                if (dim ==2 ) z2 = 0; else
                    z2 = PointsMap[it_dof->second.first].PNT[1];
                y2 = PointsMap[it_dof->second.first].Zlist[it_dof->second.second].z;
                log_file1 << x1/dbg_scale_x << ", " << y1/dbg_scale_z << ", " << z1 << ", "
                          << x2/dbg_scale_x << ", " << y2/dbg_scale_z << ", " << z2 <<  std::endl;
            }
        }
     }
     log_file1.close();
}

template<int dim>
void Mesh_struct<dim>::updateMeshElevation(DoFHandler<dim>& mesh_dof_handler,
                                           ConstraintMatrix& mesh_constraints,
                                           TrilinosWrappers::MPI::Vector& mesh_vertices,
                                           TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                                           MPI_Comm&  mpi_communicator,
                                           ConditionalOStream pcout){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);

    //dbg_meshStructInfo3D("After3D", my_rank);

    n_levels = 0;
    typename std::map<int , PntsInfo<dim> >::iterator it;

    // update level 0
    // This level has no hanging nodes and the update is straight forward
    // We simply scale the elevations between B and T according to their relative positions
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
        for (; itz != it->second.Zlist.end(); ++itz){
            if (itz->level == 0){
                itz->z = it->second.T*itz->rel_pos - it->second.B*(1.0 - itz->rel_pos);
                std::cout << itz->z << ", " << it->second.T << std::endl;
            }else{
                // just find which level this node is so that we know how many levels exist
                // for coming loops
                if (itz->level > n_levels)
                    n_levels = itz->level;
            }
        }
    }
    dbg_meshStructInfo3D("After3D", my_rank);





    // After we have finished with all updates in the z structure we have to copy the---------------------------------------
    // new values to the distributed vector
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
        for (; itz != it->second.Zlist.end(); ++itz){
            if (distributed_mesh_vertices.in_local_range(static_cast<unsigned int >(itz->dof))){
                distributed_mesh_vertices[static_cast<unsigned int >(itz->dof)] = itz->z;
            }
        }
    }
    // updates the elevations to the constraint nodes --------------------------
    mesh_constraints.distribute(distributed_mesh_vertices);
    mesh_vertices = distributed_mesh_vertices;

    //move the actual vertices ------------------------------------------------

    // for debuging just print the cell mesh
    const std::string mesh_file_name = ("mesh_after_" +
                                       Utilities::int_to_string(my_rank+1, 4) +
                                       ".dat");
    std::ofstream mesh_file;
    mesh_file.open((mesh_file_name.c_str()));

    typename DoFHandler<dim>::active_cell_iterator
    cell = mesh_dof_handler.begin_active(),
    endc = mesh_dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_artificial() == false){
            for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                Point<dim> &v=cell->vertex(vertex_no);
                for (unsigned int dir=0; dir < dim; ++dir){
                    v(dir) = mesh_vertices(cell->vertex_dof_index(vertex_no, dir));
                    if (dir == dim-1)
                        mesh_file << v(dir)/dbg_scale_z << ", ";
                    else
                        mesh_file << v(dir)/dbg_scale_x << ", ";
                }
                if (dim == 2)
                    mesh_file << 0 << ", ";
            }
            mesh_file << std::endl;
        }
    }
    mesh_file.close();

    return;


}

template <int dim>
void Mesh_struct<dim>::dbg_set_scales(double xscale, double zscale){
    dbg_scale_x = xscale;
    dbg_scale_z = zscale;
}

template <int dim>
void Mesh_struct<dim>::set_id_above_below(){
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        it->second.set_ids_above_below();
    }
}

template  <int dim>
void Mesh_struct<dim>::make_dof_ij_map(){
    dof_ij.clear();
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it)
        for (unsigned int k = 0; k < it->second.Zlist.size(); ++k){
            dof_ij[it->second.Zlist[k].dof] = std::pair<int,int> (it->first,k);
        }
}

template <int dim>
void Mesh_struct<dim>::compute_initial_elevations(MyFunction<dim, dim-1> top_function,
                                                  MyFunction<dim, dim-1> bot_function,
                                                  std::vector<double>& vert_discr){
    std::vector<double>uniform_dist = linspace(0.0, 1.0, vert_discr.size());

    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        double top = top_function.value(it->second.PNT);
        double bot = bot_function.value(it->second.PNT);
        it->T = top;
        it->B = bot;
    }
}

template <int dim>
void Mesh_struct<dim>::update_z(int level, MPI_Comm &mpi_communicator){

    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        for (unsigned int j = 0; j < it->second.Zlist.size(); ++j){

        }

    }
}

#endif // MESH_STRUCT_H
