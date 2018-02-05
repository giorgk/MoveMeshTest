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

#include <algorithm>

#include "zinfo.h"
#include "pnt_info.h"
#include "cgal_functions.h"
#include "my_functions.h"
#include "mpi_help.h"
#include "helper_functions.h"
#include "polygon_outline.h"

//! custom struct to hold data
template <int dim>
struct trianode {
    Point <dim> pnt;
    int dof;
    int hang;
    int spi; // support_point_index
    std::map<int,int> c_pnt;// dofs of points connected to that node
    int isTop;
    int isBot;
    std::vector<types::global_dof_index> cnstr_nd;
};


struct PntIndices{
    int Zind;
    int XYind;
    int isNew;
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

    //! Adds a new point in the structure. If the point exists adds the z coordinate only and returns
    //! the id of the existing point. if the point doesnt exist creates a new point and returns the new id.
    PntIndices add_new_point(Point<dim-1>, Zinfo zinfo);

    //! Checks if the point already exists in the mesh structure
    //! If the point exists it returns the id of the point in the #CGALset
    //! otherwise returns -9;
    int check_if_point_exists(Point<dim-1> p);

    /*!
     * \brief updateMeshstruct is the heart of this class. For a given parallel triangulation updates the existing
     * points or creates new ones.
     *
     * The method first loops through the locally owned cells and extracts the coordinates and dof for each node
     * which stores it to #distributed_mesh_vertices. Then keeps in a custom map information for each node of the
     * triangulation such as : dof, level (level is set only once when the node is first created), whether is a hanging
     * node, and a list of connections with othe nodes, where for each connection the dof, level and hanging information is also stored.
     * NOTE: the connected nodes may be more than what they actually are:
     *
     *          a       d
     * -----------------
     * |___|___|c      |
     * |   |   |       |
     * -----------------
     *         b        e
     *
     *
     * In the example above node a would appear to have connections with d b and c. While this is not correct doesnt seem to
     * influence the algorithm because the hanging nodes have always the correct number of connections
     *
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
                         ConditionalOStream pcout,
                         std::string prefix);

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
                             ConditionalOStream pcout,
                             std::string prefix);

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

    //! This creates the #dof_ij map. In addition removes points in the struct that are not used
    void make_dof_ij_map();

    //! This method calculates the top and bottom elevation on the points of the #PointsMap
    //! This should be called on the initial grid before any refinement
    void compute_initial_elevations(MyFunction<dim, dim-1> top_function,
                                    MyFunction<dim, dim-1> bot_function,
                                    std::vector<double>& vert_discr);

    //! This method sets the scales #dbg_scale_x and #dbg_scale_z for debug plotting using softwares like houdini
    void dbg_set_scales(double xscale, double zscale);


    void move_vertices(DoFHandler<dim>& mesh_dof_handler,
                       TrilinosWrappers::MPI::Vector& mesh_vertices,
                       unsigned int my_rank,
                       std::string prefix);

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
    dbg_scale_z = 20;
}

template <int dim>
PntIndices Mesh_struct<dim>::add_new_point(Point<dim-1>p, Zinfo zinfo){
    PntIndices outcome;
    outcome.XYind = -99;
    outcome.Zind = -99;
    outcome.isNew = -99;

    //if (zinfo.dof == 189)
    //    std::cout << "SO FAR SO GOOD" << std::endl;

    //if (zinfo.dof < 0)
    //    std::cerr << "You attepmt to add a point with negative dof" << std::endl << std::flush;

    // First search for the XY location in the structure
    int id = check_if_point_exists(p);

    //if (zinfo.dof == 189)
    //    std::cout << "id is " << id << std::endl;

    if ( id < 0 ){
        // this is a new point and we add it to the map
        PntsInfo<dim> tempPnt(p, zinfo);
        tempPnt.find_id = _counter;
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
        outcome.isNew = 1;
        _counter++;
    }else if (id >=0){
        typename std::map<int, PntsInfo<dim> >::iterator it = PointsMap.find(id);
        //if (zinfo.dof == 189){
        //    if (it == PointsMap.end())
        //        std::cout << "NO WAY" << std::endl;
        //    else
        //        std::cout << it->second.PNT[0] << std::endl;
        //}
        it->second.add_Zcoord(zinfo, z_thres);
        outcome.XYind = it->first;
        outcome.isNew = 0;
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
void Mesh_struct<dim>::updateMeshStruct(DoFHandler<dim>& mesh_dof_handler,
                                       FESystem<dim>& mesh_fe,
                                       ConstraintMatrix& mesh_constraints,
                                       IndexSet& mesh_locally_owned,
                                       IndexSet& mesh_locally_relevant,
                                       TrilinosWrappers::MPI::Vector& mesh_vertices,
                                       TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                                       MPI_Comm&  mpi_communicator,
                                       ConditionalOStream pcout,
                                       std::string prefix){
    // Use this to time the operation. Note that this is a very expensive operation but nessecary
    std::clock_t begin_t = std::clock();
    // get the rank and processor id just for output display
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    // make sure all processors start together
    MPI_Barrier(mpi_communicator);
    reset(); // delete all
    MPI_Barrier(mpi_communicator);

    const MappingQ1<dim> mapping;

    pcout << "Distribute mesh dofs..." << mesh_dof_handler.n_dofs() << std::endl << std::flush;

    mesh_dof_handler.distribute_dofs(mesh_fe);
    //pcout << "dofs 1" << mesh_dof_handler.n_dofs() << std::endl << std::flush;
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

    // Make a list of points in x-y that
    std::vector<std::vector<Point<dim-1> > > pointsXY(n_proc);
    std::vector<std::vector<ine_Point2 > > pointsXYcgal(n_proc);

    pcout << "Update XYZ structure...for: " << prefix  << std::endl << std::flush;
    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell = mesh_dof_handler.begin_active(),
        endc = mesh_dof_handler.end();
    //std::cout << "Rank: " << my_rank << " MADE IT to 00000 for " << prefix << std::endl;
    MPI_Barrier(mpi_communicator);
    //int dbg_cnt =0;
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            bool top_cell = false;
            bool bot_cell = false;

            if (cell->neighbor_index(GeometryInfo<dim>::faces_per_cell-2) < 0){
                //std::cout << "Is bottom" << std::endl;
                bot_cell = true;
            }
            if (cell->neighbor_index(GeometryInfo<dim>::faces_per_cell-1) < 0){
                //std::cout << "Is top" << std::endl;
                top_cell = true;
            }

            //if (my_rank == 0){
            //    std::cout << "++++cell " << dbg_cnt++ << "start+++++" << std::endl;
            //}
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
                temp.hang = mesh_constraints.is_constrained(current_dofs[dim-1]);
                temp.cnstr_nd.push_back(current_dofs[dim-1]);
                mesh_constraints.resolve_indices(temp.cnstr_nd);
                //if (current_dofs[dim-1] == 519)
                //    std::cout <<"@@@@@@@@@@@@@@@@@@@@@@@@ " << temp.cnstr_nd.size() << "##################" << std::endl;
                temp.spi = spi[dim-1];
                temp.isBot = 0;
                temp.isTop = 0;
                if (bot_cell){
                    if (idof < GeometryInfo<dim>::vertices_per_cell/2){
                        temp.isBot = 1;
                        //std::cout << "bottom point" << std::endl;
                    }
                }
                if (top_cell){
                    if (idof >= GeometryInfo<dim>::vertices_per_cell/2){
                        temp.isTop = 1;
                        //std::cout << "top point" << std::endl;
                    }
                }
                curr_cell_info[idof] = temp;
                //if (my_rank == 0){
                //    std::cout << "(" << temp.pnt[0] << ", " << temp.pnt[1] << "), ";
                //}
            }
            //if (my_rank == 0){
            //    std::cout << std::endl;
            //    std::cout << "       Finish first part    " << std::endl;
            //}

            typename std::map<int, trianode<dim> >::iterator it;
            for (it = curr_cell_info.begin(); it != curr_cell_info.end(); ++it){
                //if (my_rank == 0 && dbg_cnt >= 62){
                //    std::cout  << "a: " << it->first << ", " << it->second.dof << std::endl;
                //}
                // get the nodes connected with this one
                std::vector<int> id_conn = get_connected_indices<dim>(it->first);

                //if (my_rank == 0 && dbg_cnt >= 62 && it->first == 3 && it->second.dof == 189 )
                //    std::cout<< "Made it here I " << std::endl;

                // create a vector of the points conected with this one
                std::vector<int> connectedNodes;
                for (unsigned int i = 0; i < id_conn.size(); ++i){
                    connectedNodes.push_back(curr_cell_info[id_conn[i]].dof);
                }

                // create a vector of ints to hold the nodes that this node depends on if its constrained
                std::vector<int> temp_cnstr;
                for (unsigned int ii = 0; ii < it->second.cnstr_nd.size(); ++ii){
                    temp_cnstr.push_back(it->second.cnstr_nd[ii]);
                }

                // Now create a zinfo variable
                Zinfo zinfo(it->second.pnt[dim-1], it->second.dof, temp_cnstr, it->second.isTop, it->second.isBot, connectedNodes);

                // and a point
                Point<dim-1> ptemp;
                for (unsigned int d = 0; d < dim-1; ++d)
                    ptemp[d] = it->second.pnt[d];

                //if (my_rank == 0 && dbg_cnt >= 62 && it->first == 3 && it->second.dof == 189 )
                //    std::cout<< "Made it here III " << std::endl;

                // Try to add it in the structure
                //std::cout <<"Zdof: " << zinfo.dof << std::endl;
                PntIndices id_in_map = add_new_point(ptemp, zinfo); // MAYBE WE DONT NEED TO RETURN ANYTHING
            }
        }
    }

    //std::cout << "Rank: " << my_rank << " MADE IT to AA for " << prefix << std::endl;
    MPI_Barrier(mpi_communicator);
    make_dof_ij_map();

    //Outlines[my_rank].PrintPolygons(my_rank);
    //if (my_rank == 0){
    //    std::vector<int> ints;
    //    std::vector<double> dbls;
    //    Outlines[my_rank].serialize();
    //}


    MPI_Barrier(mpi_communicator);
    distributed_mesh_vertices.compress(VectorOperation::insert);
    MPI_Barrier(mpi_communicator);

    //dbg_meshStructInfo2D("before2D", my_rank);
    dbg_meshStructInfo3D("before3D_Struct_" + prefix + "_", my_rank);


    if (n_proc > 1){
        pcout << "exchange vertices between processors..." << std::endl << std::flush;


        // Each processor will make a list of the XY coordinates
        std::vector<std::vector<double> > Xcoords(n_proc);
        std::vector<std::vector<double> > Ycoords(n_proc);
        // and a list of the keys where the coordinates correspond
        //std::vector<std::vector<int> > key_map(n_proc);
        std::vector<int> n_points_per_proc(n_proc);
        typename std::map<int ,  PntsInfo<dim> >::iterator it, itf;
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            //key_map[my_rank].push_back(it->first);
            Xcoords[my_rank].push_back(it->second.PNT[0]);
            if (dim == 3)
                Ycoords[my_rank].push_back(it->second.PNT[1]);

        }


        Send_receive_size(static_cast<unsigned int>(Xcoords[my_rank].size()), n_proc, n_points_per_proc, mpi_communicator);
        //Sent_receive_data<int>(key_map, n_points_per_proc, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(Xcoords, n_points_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(Ycoords, n_points_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);


        // Each processor knows what xy points the other processor have
        // Lets suppose that I'm processor 0 and talk with the other processors
        // -I'm processor 0 and I will loop though the points (XYcoords) that the other processors have sent me.
        //  If I find any point in theses lists I received from the other processors in my PointMap
        //  I will flag it as shared and I'll add the id of the processor that I share the point with.
        for (unsigned int i = 0; i < n_proc; ++i){
            if (i == my_rank)
                continue;
            for (unsigned int j = 0; j < Xcoords[i].size(); ++j){
                Point<dim-1> p;
                p[0] = Xcoords[i][j];
                if (dim == 3)
                    p[1] = Ycoords[i][j];

                int id = check_if_point_exists(p);
                if (id >=0){
                    itf = PointsMap.find(id);
                    if (itf != PointsMap.end()){
                        itf->second.shared_proc.push_back(i);
                        //itf->second.key_val_shared_proc.push_back(key_map[i][j]);
                    }
                }
            }
        }

        // Now that I know which XY points I share with the other processors I will make a list
        // of the dofs that each of these points possess
        std::vector<int> n_dofZ_per_proc(n_proc);
        std::vector<std::vector<int> > dofZ(n_proc);
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            if (it->second.shared_proc.size() > 0){// if this point is shared with other processors SEND:
                dofZ[my_rank].push_back(it->second.shared_proc.size()); // ---- the number of processors that share the point
                for (unsigned int k = 0; k < it->second.shared_proc.size(); ++k){
                    dofZ[my_rank].push_back(it->second.shared_proc[k]);// ---- the processor id that needs this point
                }
                dofZ[my_rank].push_back(it->second.Zlist.size()); // ---- the number the dofs under the XY location
                for (unsigned int j = 0; j < it->second.Zlist.size(); ++j){
                    dofZ[my_rank].push_back(it->second.Zlist[j].dof); // ---- the dof itself
                }
            }
        }


        //print_size_msg<double>(Xcoords, my_rank);
        //if (my_rank == 0){
        //    for (unsigned int i = 0; i < dofZ[my_rank].size(); ++i)
        //        std::cout << "-- " << dofZ[my_rank][i] << std::endl;
        //}
        //return;

        //Exchange the dofs
        Send_receive_size(static_cast<unsigned int>(dofZ[my_rank].size()), n_proc, n_dofZ_per_proc, mpi_communicator);
        Sent_receive_data<int>(dofZ, n_dofZ_per_proc, my_rank, mpi_communicator, MPI_INT);

        //print_size_msg<int>(dofZ, my_rank);
        //return;

        // Now I will loop though the dofs that the other processors have sent me
        // If any of the other processor have told me that I share points with them
        // I'll go though the dofs and pick the ones I dont have in my list and put them
        // in a dof request vector
        std::vector<std::vector<int> > Request_dof(n_proc);
        std::map<int,int> unique_list_req_dofs;// Use this map to make sure we request the node only from one processor
        for (unsigned int i = 0; i < n_proc; ++i){
            if (i == my_rank)
                continue;
            std::vector<int> temp_dof_request;
            unsigned int i_cnt = 0;
            while(true){
                int Nproc_share = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                bool isthisme = false;
                for (int k = 0; k < Nproc_share; ++k){
                    int iproc = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                    if (iproc == static_cast<int>(my_rank)){ // if the shared processor is me I should check which dofs I am missing
                        isthisme = true;
                    }
                }
                int Ndofs = get_v<int>(dofZ, i, i_cnt); i_cnt++;// Number of dofs in this key
                for (int k = 0; k < Ndofs; ++k){
                    int temp_dof = get_v<int>(dofZ, i, i_cnt); i_cnt++;
                    if (isthisme){
                        if (dof_ij.find(temp_dof) == dof_ij.end()){// if I dont have this in my list I'll request the info for it
                            temp_dof_request.push_back(temp_dof);
                            if (unique_list_req_dofs.find(temp_dof) == unique_list_req_dofs.end()){
                                unique_list_req_dofs.insert(std::pair<int,int>(temp_dof, temp_dof));
                            }
                        }
                    }
                }
                if (i_cnt >= dofZ[i].size())
                    break;
            }
            Request_dof[my_rank].push_back(i); // the processor we request the data from
            Request_dof[my_rank].push_back(static_cast<int>(temp_dof_request.size())); //the total number number of dofs that we want from this processor
            for (unsigned int k = 0; k < temp_dof_request.size(); ++k)
                Request_dof[my_rank].push_back(temp_dof_request[k]);
        }

        //if (my_rank == 3){
        //    for (unsigned int i = 0; i < Request_dof[my_rank].size(); ++i)
        //        std::cout << "-- " << Request_dof[my_rank][i] << std::endl;
        //}
        //return;

        // Sent out my requests
        std::vector<int> n_request_per_proc(n_proc);
        Send_receive_size(static_cast<unsigned int>(Request_dof[my_rank].size()), n_proc, n_request_per_proc, mpi_communicator);
        Sent_receive_data<int>(Request_dof, n_request_per_proc, my_rank, mpi_communicator, MPI_INT);

        //print_size_msg<int>(Request_dof, my_rank);
        //return;

        // Now I have to loop through the other processors requests.
        // If I find that any processor needs data from me I'll pack the data into two
        // serialized vectors of type int and double for the Zcoord

        std::vector<std::vector<int>> int_data(n_proc);
        std::vector<std::vector<double>> dbl_data(n_proc);
        std::map<int, std::pair<int, int>>::iterator it_dof;
        for (unsigned int i = 0; i < n_proc; ++i){
            if (i == my_rank)
                continue;

            unsigned int i_cnt = 0;
            int Ndofs_2_sent = 0;
            std::vector<int> temp_int_data;
            temp_int_data.clear();
            while (true){
                int proc2send = get_v<int>(Request_dof, i, i_cnt); i_cnt++;// This is the processor that should sent the data to i
                int Ndofs_request = get_v<int>(Request_dof, i, i_cnt); i_cnt++;// This is how many data i has requested
                for (int j = 0; j < Ndofs_request; ++j){
                    int req_dof = get_v<int>(Request_dof, i, i_cnt); i_cnt++; // This is the requested dof
                    if (proc2send == static_cast<int>(my_rank)){// If I am supposed to send this dof I have to pack its data as follows:
                        it_dof = dof_ij.find(req_dof);
                        if (it_dof != dof_ij.end()){
                            Ndofs_2_sent++;
                            Zinfo zinf = PointsMap[it_dof->second.first].Zlist[it_dof->second.second];
                            if (zinf.dof == req_dof){
                                temp_int_data.push_back(zinf.dof);                  // 1) The dof
                                if (zinf.isTop == 1)                                // 2) a flag about the top/ bottom status of the dof
                                    temp_int_data.push_back(1);
                                else if (zinf.isBot == 1)
                                    temp_int_data.push_back(2);
                                else
                                    temp_int_data.push_back(0);

                                temp_int_data.push_back(static_cast<int>(zinf.dof_conn.size()));      // 3) The number of connections for this point
                                for (unsigned int icn = 0; icn <  zinf.dof_conn.size(); ++icn){
                                    temp_int_data.push_back(zinf.dof_conn[icn]);          // 4) The connected dofs
                                }
                                temp_int_data.push_back(static_cast<int>(zinf.cnstr_nds.size()));     // 5) The number of constraints for this point
                                for (unsigned int jj = 0; jj < zinf.cnstr_nds.size(); ++jj){
                                    temp_int_data.push_back(zinf.cnstr_nds[jj]);    // 6) The constraint dofs
                                }

                                // The coordinates
                                dbl_data[my_rank].push_back(PointsMap[it_dof->second.first].PNT[0]);
                                if (dim == 3)
                                    dbl_data[my_rank].push_back(PointsMap[it_dof->second.first].PNT[1]);
                                dbl_data[my_rank].push_back(zinf.z);                // ON a separate vector send the Zcoord

                            }else{
                                std::cerr << "There is a mismatch between dofs" << std::endl;
                            }
                        }
                    }
                }
                if (i_cnt >= Request_dof[i].size()){
                    break;
                }
            }

            // for each processor we loop add the data
            int_data[my_rank].push_back(static_cast<int>(i)); // The processor id that asked for the data
            int_data[my_rank].push_back(Ndofs_2_sent); // The number of dofs nodes that this processor asked
            for (unsigned int kk = 0; kk < temp_int_data.size(); ++kk)
                int_data[my_rank].push_back(temp_int_data[kk]);
        }



        std::vector<int> n_ints_per_proc(n_proc);
        std::vector<int> n_dbls_per_proc(n_proc);
        Send_receive_size(static_cast<unsigned int>(int_data[my_rank].size()), n_proc, n_ints_per_proc, mpi_communicator);
        Send_receive_size(static_cast<unsigned int>(dbl_data[my_rank].size()), n_proc, n_dbls_per_proc, mpi_communicator);
        Sent_receive_data<int>(int_data, n_ints_per_proc, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(dbl_data, n_dbls_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);

        //print_size_msg<int>(int_data, my_rank);
        //print_size_msg<double>(dbl_data, my_rank);
        //return;

        // Finally I'll loop trhough the dofs that the porcessors have sent and pick the ones I had asked
        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (i_proc == my_rank)
                continue;
            unsigned int i_cnt = 0;
            unsigned int d_cnt = 0;
            while (true){
                int proc_ask_data = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                int Ndofs = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;

                for (int i = 0; i < Ndofs; ++i){
                    int dof = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    int top_flag = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    int n_conn = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    std::vector<int> conn;
                    for (int j = 0; j < n_conn; ++j){
                        conn.push_back(get_v<int>(int_data, i_proc, i_cnt)); i_cnt++;
                    }
                    int n_cnstr = get_v<int>(int_data, i_proc, i_cnt); i_cnt++;
                    std::vector<int> cnstr;
                    for (int j = 0; j < n_cnstr; ++j){
                        cnstr.push_back(get_v<int>(int_data, i_proc, i_cnt)); i_cnt++;
                    }

                    double x = get_v<double>(dbl_data, i_proc, d_cnt); d_cnt++;
                    double y = get_v<double>(dbl_data, i_proc, d_cnt); d_cnt++;
                    double z = get_v<double>(dbl_data, i_proc, d_cnt); d_cnt++;

                    if (proc_ask_data == static_cast<int>(my_rank)){// If I have asked for the dof, I'll add it to my points map
                        int istop = 0;
                        int isbot = 0;
                        if (top_flag == 1)
                            istop = 1;
                        else if (top_flag == 2)
                            isbot = 1;

                        Zinfo newZ(z,dof,cnstr,istop,isbot,conn);
                        Point<dim-1> ptemp;
                        ptemp[0] = x;
                        if (dim == 3)
                            ptemp[1] = y;
                        PntIndices pnt_ind = add_new_point(ptemp,newZ);
                    }
                }
                if (i_cnt >= int_data[i_proc].size() || d_cnt >= dbl_data[i_proc].size())
                    break;
            }
        }
    }//if (n_proc > 1)

    // IN the new code this may not needed. However it is needed for debuging
    make_dof_ij_map();

    set_id_above_below();
    dbg_meshStructInfo3D("After3D_Struct_" + prefix + "_", my_rank);

    std::clock_t end_t = std::clock();
    double elapsed_secs = double(end_t - begin_t)/CLOCKS_PER_SEC;
    //std::cout << "====================================================" << std::endl;
    std::cout << "I'm rank " << my_rank << " and spend " << elapsed_secs << " sec on Updating XYZ" << std::endl;
    //std::cout << "====================================================" << std::endl;
    MPI_Barrier(mpi_communicator);
}

template <int dim>
void Mesh_struct<dim>::reset(){
    _counter = 0;
    PointsMap.clear();
    dof_ij.clear();
    CGALset.clear();
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
                      << std::setw(15) << itz->cnstr_nds.size() << ", "
                      << std::setw(15) << itz->isTop << ", "
                      << std::setw(15) << itz->isBot << ", "
                      << std::setw(15) << itz->dof_above  << ", "
                      << std::setw(15) << itz->dof_below << ", "
                      << std::setw(15) << itz->dof_top  << ", "
                      << std::setw(15) << itz->dof_bot << ", "
                      << std::setw(15) << itz->rel_pos  << ", "
                      << std::setw(15) << itz->hanging << ", "
                      << std::setw(15) << it->second.T << ", "
                      << std::setw(15) << it->second.B << ", "
                      << std::setw(5) << it->second.have_to_send
                      << std::endl;


             std::map<int, int >::iterator itt;
             for (unsigned int i = 0; i < itz->dof_conn.size(); ++i){
                 int a,b;
                 if (itz->dof < itz->dof_conn[i]){
                     a = itz->dof;
                     b = itz->dof_conn[i];
                 }
                 else{
                     a = itz->dof_conn[i];
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
                log_file1 << x1/dbg_scale_x << ", " << y1/dbg_scale_z << ", " << z1/dbg_scale_x << ", "
                          << x2/dbg_scale_x << ", " << y2/dbg_scale_z << ", " << z2/dbg_scale_x <<  std::endl;
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
                                           ConditionalOStream pcout,
                                           std::string prefix){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);

    typename std::map<int , PntsInfo<dim> >::iterator it;
    std::map<int,std::pair<int,int> >::iterator it_dm; // iterator for dof_ij

    int dbg_iter = 0;
    while (true){
        int n_not_set = 0;
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
            for (; itz != it->second.Zlist.end(); ++itz){
                //std::cout << "Rank [" << my_rank << "]x: " << it->second.PNT << " z: (" << itz->dof << "): " << itz->z << std::endl;
                if (itz->isZset)
                    continue;

                if (itz->isTop){// The nodes on the top lsurface gets their values directly
                    itz->z = it->second.T;
                    itz->isZset = true;
                }
                else if (itz->isBot){ // Same for the nodes on the bottom
                    itz->z = it->second.B;
                    itz->isZset = true;
                }
                else if (itz->hanging){
                    // if the node is constraint we get a list of ids that this node depends on
                    // and average their values only if all of them have been set at this iteration
                    // The boolean is_complete gets false if any of the nodes has not been set this iteration
                    bool is_complete = true;
                    double newz = 0;
                    double cntz = 0;

                    for (unsigned int ii = 0; ii < itz->cnstr_nds.size(); ++ii){
                        //find the ij indices in the PointMap structure
                        it_dm = dof_ij.find(itz->cnstr_nds[ii]);
                        if (it_dm != dof_ij.end()){
                            if (PointsMap[it_dm->second.first].Zlist[it_dm->second.second].isZset){
                                newz += PointsMap[it_dm->second.first].Zlist[it_dm->second.second].z;
                                cntz = cntz + 1.0;
                            }
                            else{
                                is_complete = false;
                                break;
                            }
                        }
                    }
                    if (is_complete){
                        if (cntz >0){
                            itz->z = newz / cntz;
                            itz->isZset = true;
                        }
                        else
                            std::cout << "Rank: " << my_rank << " has constrained dof " << itz->dof << " with 0 constraint nodes" << std::endl;
                    }

                }
                else{
                    if (it->second.Zlist[itz->id_bot].isZset == true && it->second.Zlist[itz->id_top].isZset == true){
                        if (it->second.Zlist[itz->id_bot].isZset == true && it->second.Zlist[itz->id_top].isZset == true){
                            itz->z = it->second.Zlist[itz->id_top].z*itz->rel_pos + it->second.Zlist[itz->id_bot].z*(1.0 - itz->rel_pos);
                            itz->isZset = true;
                        }
                    }
                }
                if (distributed_mesh_vertices.in_local_range(static_cast<unsigned int >(itz->dof))){
                    if (itz->isZset)
                        distributed_mesh_vertices[static_cast<unsigned int >(itz->dof)] = itz->z;
                    else{
                        //std::cout << "R: " << my_rank << " dof " << itz->dof << std::endl;
                        n_not_set++;
                    }
                }
            }// loop Z points
        }// loop x-y points
        std::cout << n_not_set << std::endl;
        if (n_not_set == 0)
            break;
    }
    std::cout << "Rank " << my_rank << " has converged" << std::endl;
/*
    int n_levels = 0;
    typename std::map<int , PntsInfo<dim> >::iterator it;
    std::map<int,std::pair<int,int> >::iterator it_con, it_df;
    // update level 0
    // This level has no hanging nodes and the update is straightforward
    // We simply scale the elevations between B and T according to their relative positions.
    // In addition we can set the elevations on the nodes that belong to the top or bottom faces.
    // These nodes are the ones that have no connections below or above respectively and
    // they are not hanging nodes
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
        for (; itz != it->second.Zlist.end(); ++itz){
            if (itz->level == 0){
                //std::cout << "Zold: " << itz->z << ", r: " << itz->rel_pos << ", T: " << it->second.T << std:: endl;
                itz->z = it->second.T*itz->rel_pos + it->second.B*(1.0 - itz->rel_pos);
                itz->isZset = true;
                //std::cout << "Znew: " << itz->z << std::endl;
            }else{
                // just find which level this node is so that we know how many levels exist
                // for the coming loop
                if (itz->level > n_levels)
                    n_levels = itz->level;

                if (itz->isTop == 1){
                    itz->z = it->second.T;
                    itz->isZset = true;
                }

                if (itz->isBot == 1){
                    itz->z = it->second.B;
                    itz->isZset = true;
                }
            }

            // In addition we will set the levels of the nodes connected to each point
            for (it_con = itz->dof_conn.begin(); it_con != itz->dof_conn.end(); ++ it_con){
                it_df = dof_ij.find(it_con->first);
                if (it_df != dof_ij.end()){
                    it_con->second.first = PointsMap[it_df->second.first].Zlist[it_df->second.second].level;
                }else{
                    //std::cerr << "I'm proc " << my_rank << " and I couldnt find the node with dof: " << it_con->first << " in my dof_ij map" << std::endl;
                }
            }
            if (itz->isZset){
                if (distributed_mesh_vertices.in_local_range(static_cast<unsigned int >(itz->dof))){
                    distributed_mesh_vertices[static_cast<unsigned int >(itz->dof)] = itz->z;
                }
            }
        }
    }


    std::map<int,std::pair<int,int> >::iterator it_c;// iterator for connected points (dof,<level,hanging>)
    std::map<int,std::pair<int,int> >::iterator it_dm; // iterator for dof_ij
    while (true){
        int n_not_set = 0;
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
            for (; itz != it->second.Zlist.end(); ++itz){
                if (itz->isZset)
                    continue;
                if (itz->connected_above && itz->connected_below){
                    if (it->second.Zlist[itz->id_bot].isZset == true && it->second.Zlist[itz->id_top].isZset == true){
                        itz->z = it->second.Zlist[itz->id_top].z*itz->rel_pos + it->second.Zlist[itz->id_bot].z*(1.0 - itz->rel_pos);
                        itz->isZset = true;
                    }
                }
                else if ((itz->connected_below && !itz->connected_above) || (!itz->connected_below && itz->connected_above)){
                    std::cout << "Z: " << itz->dof << std::endl;
                    bool is_complete = true;
                    double newz = 0;
                    double cntz = 0;
                    bool use_samelevel = true;
                    // find out the levels of the connected nodes
                    for (it_c = itz->dof_conn.begin(); it_c != itz->dof_conn.end(); ++it_c){
                        if (it_c->first == itz->id_below || it_c->first == itz->id_above)
                            continue;
                        if (it_c->second.first < itz->level){
                            use_samelevel = false;
                            break;
                        }
                    }
                    std::vector< types::global_dof_index > indices;
                    indices.push_back(itz->dof);
                    mesh_constraints.resolve_indices(indices);
                    // loop though the connections
                    for (it_c = itz->dof_conn.begin(); it_c != itz->dof_conn.end(); ++it_c){
                        std::cout << "c: " << it_c->first << std::endl;
                        if (it_c->first != itz->id_below && it_c->first != itz->id_above){// we skip the connection with the node below or above
                            if (!use_samelevel)
                                if (it_c->second.first >= itz->level)
                                    continue;
                            it_dm = dof_ij.find(it_c->first);
                            if (it_dm != dof_ij.end()){
                                if (PointsMap[it_dm->second.first].Zlist[it_dm->second.second].isZset){
                                    newz += PointsMap[it_dm->second.first].Zlist[it_dm->second.second].z;
                                    cntz = cntz + 1.0;
                                }
                                else{
                                    is_complete = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (is_complete){
                        itz->z = newz / cntz;
                        itz->isZset = true;
                    }
                }

                if (distributed_mesh_vertices.in_local_range(static_cast<unsigned int >(itz->dof))){
                    if (itz->dof == 2561)
                        std::cout << my_rank << std::endl;
                    if (itz->isZset)
                        distributed_mesh_vertices[static_cast<unsigned int >(itz->dof)] = itz->z;
                    else
                        n_not_set++;
                }
            }
        }
        std::cout << n_not_set << std::endl;
        if (n_not_set == 0)
            break;
    }
*/
    MPI_Barrier(mpi_communicator);
/*
    // However for all other levels we have to do the z calculations in a specific order.
    // For each level we loop through the nodes twice.
    // The first time we will update the elevations of the hanging nodes that have no
    // connection with a node above or below. If the hanging node has both connections
    // with the nodes above and below we can calculate safely its elevation on the second loop.

    for (int i_lvl = 1; i_lvl <= n_levels; ++i_lvl){
        // loop 1: hanging nodes that depend on previous level nodes
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
            for (; itz != it->second.Zlist.end(); ++itz){
                if (itz->level == i_lvl && itz->isZset == false){
                    if (itz->hanging){
                        if (!itz->connected_above || !itz->connected_below){
                            //std::cout << "-------x: " << it->second.PNT[0] << ", dof: " << itz->dof << " ---------" << std::endl;
                            //std::cout << "Zold: " << itz->z << ", r: " << itz->rel_pos << std::endl;
                            // loop through its connencetd points and average the z value
                            // of those that have lower level than this node level (i_lvl)
                            double newz = 0; double cntz = 0;
                            std::map<int,std::pair<int,int> >::iterator it_c;// iterator for connected points (dof,<level,hanging>)
                            std::map<int,std::pair<int,int> >::iterator it_dm; // iterator for dof_ij
                            for (it_c = itz->dof_conn.begin(); it_c != itz->dof_conn.end(); ++it_c){
                                if (it_c->second.first < i_lvl && (it_c->first != itz->id_above && it_c->first != itz->id_below)){
                                    if (itz->dof == 2345)
                                        std::cout << "AA 2345 " << it_c->first << std::endl;
                                    it_dm = dof_ij.find(it_c->first);
                                    if (it_dm != dof_ij.end()){
                                        //std::cout << "dof: " << it_c->first << ", z: " << PointsMap[it_dm->second.first].Zlist[it_dm->second.second].z << std::endl;
                                        newz += PointsMap[it_dm->second.first].Zlist[it_dm->second.second].z;
                                        cntz += 1.0;
                                    }
                                    else{
                                        std::cerr << "Rank " << my_rank << ": For the node: " << itz->dof << " I coudlnt find the "
                                                  << it_c->first << " which is supposed to be connected" << std::endl;
                                    }
                                }
                            }
                            if (cntz < 1){
                                std::cerr << " no conections found for the node with dof: " << itz->dof << std::endl;
                            }
                            else{
                                itz->z = newz / cntz;
                                itz->isZset = true;
                                //std::cout << "Znew: " << itz->z << std::endl;
                            }
                        }
                    }
                }
            }
        }

        // loop 2: hanging nodes that depend on hanging nodes of the same level
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
            for (; itz != it->second.Zlist.end(); ++itz){
                if (itz->level == i_lvl  && itz->isZset == false){
                    if (itz->hanging){
                        if (!itz->connected_above || !itz->connected_below){
                            double newz = 0; double cntz = 0;
                            std::map<int,std::pair<int,int> >::iterator it_c;// iterator for connected points (dof,<level,hanging>)
                            std::map<int,std::pair<int,int> >::iterator it_dm; // iterator for dof_ij
                            for (it_c = itz->dof_conn.begin(); it_c != itz->dof_conn.end(); ++it_c){
                                if (it_c->second.first == i_lvl && (it_c->first != itz->id_above && it_c->first != itz->id_below)){
                                    if (itz->dof == 2345)
                                        std::cout << "BB 2345 " << it_c->first << std::endl;
                                    it_dm = dof_ij.find(it_c->first);
                                    if (it_dm != dof_ij.end()){
                                        newz += PointsMap[it_dm->second.first].Zlist[it_dm->second.second].z;
                                        cntz += 1.0;
                                    }
                                    else{
                                        std::cerr << "Rank " << my_rank << ": For the node: " << itz->dof << " I coudlnt find the "
                                                  << it_c->first << " which is supposed to be connected" << std::endl;
                                    }
                                }
                            }
                            if (cntz < 1){
                                std::cerr << " no conections found for the node with dof: " << itz->dof << std::endl;
                            }
                            else{
                                itz->z = newz / cntz;
                                itz->isZset = true;
                                //std::cout << "Znew: " << itz->z << std::endl;
                            }
                        }
                    }
                }
            }
        }

        // loop 3: change all the remaining elevations on this level
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
            for (; itz != it->second.Zlist.end(); ++itz){
                if (itz->level == i_lvl){
                    if (itz->connected_above && itz->connected_below){
                        //std::cout << "-------x: " << it->second.PNT[0] << ", dof: " << itz->dof << " ---------" << std::endl;
                        //std::cout << "Zold: " << itz->z << ", r: " << itz->rel_pos << std::endl;
                        if (it->second.Zlist[itz->id_bot].isZset == false)
                            std::cout << "RANK " << my_rank << " has dof " << itz->dof << " will use as bot " << itz->dof_bot << " value which has not been yet set" << std::endl;

                        double zb = it->second.Zlist[itz->id_bot].z;
                        double zt;
                        //if (it->second.Zlist[itz->id_top].hanging == 0)
                        //    zt = it->second.T;
                        //else
                        if (it->second.Zlist[itz->id_top].isZset == false)
                            std::cout << "RANK " << my_rank << " has dof " << itz->dof << " will use as top " << itz->dof_top << " value which has not been yet set" << std::endl;

                        zt = it->second.Zlist[itz->id_top].z;
                        //std:: cout << "zt: " << zt << ", zb: " << zb << " dof(t,b): (" << itz->dof_top << "," << itz->dof_bot
                        //           << "), oid(t,b): " << itz->id_top << "," << itz->id_bot << ")" << std::endl;
                        itz->z = zt*itz->rel_pos + zb*(1.0 - itz->rel_pos);
                        //std::cout << "Znew: " << itz->z << std::endl;
                    }
//                    else if (itz->hanging == 0 && !itz->connected_above){
//                        itz->z = it->second.T*itz->rel_pos + it->second.Zlist[itz->id_bot].z * (1.0 - itz->rel_pos);
//                    }
//                    else if (itz->hanging == 0 && !itz->connected_below){
//                        itz->z = it->second.Zlist[itz->id_top].z * itz->rel_pos + it->second.B*(1.0 - itz->rel_pos);
//                    }
                }
            }
        }

    }// loop through levels







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
    */

    dbg_meshStructInfo3D("After3D_Elev_" + prefix + "_", my_rank);

    // The compress sends the data to the processors that owns the data
    distributed_mesh_vertices.compress(VectorOperation::insert);

    // updates the elevations to the constraint nodes --------------------------
    mesh_constraints.distribute(distributed_mesh_vertices);
    mesh_vertices = distributed_mesh_vertices;

    //move the actual vertices ------------------------------------------------
    move_vertices(mesh_dof_handler,
                  mesh_vertices,
                  my_rank, prefix);
}

template <int dim>
void Mesh_struct<dim>::move_vertices(DoFHandler<dim>& mesh_dof_handler,
                                     TrilinosWrappers::MPI::Vector& mesh_vertices,
                                     unsigned int my_rank,
                                     std::string prefix){
    // for debuging just print the cell mesh
    const std::string mesh_file_name = ("mesh_after_" + prefix + "_" +
                                        Utilities::int_to_string(my_rank+1, 4) +
                                        ".dat");

    std::ofstream mesh_file;
    mesh_file.open((mesh_file_name.c_str()));

    typename DoFHandler<dim>::active_cell_iterator
    cell = mesh_dof_handler.begin_active(),
    endc = mesh_dof_handler.end();
    double x,y,z;
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){//cell->is_artificial() == false
            for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                Point<dim> &v=cell->vertex(vertex_no);
                for (unsigned int dir=0; dir < dim; ++dir){
                    v(dir) = mesh_vertices(cell->vertex_dof_index(vertex_no, dir));
                    if (dir == 0)
                        x = v(dir)/dbg_scale_x;
                    if (dir == 1 && dim == 2){
                        y = v(dir)/dbg_scale_z;
                        z = 0;
                    }
                    if (dir == 1 && dim == 3){
                        z = v(dir)/dbg_scale_x;
                    }
                    if (dir == 2 && dim == 3)
                        y = v(dir)/dbg_scale_z;
                }
                mesh_file << x << ", " << y << ", " << z << ", ";
            }
            mesh_file << std::endl;
        }
    }
    mesh_file.close();
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
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        for (unsigned int k = 0; k < it->second.Zlist.size(); ++k){
            dof_ij[it->second.Zlist[k].dof] = std::pair<int,int> (it->first,k);
        }
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


#endif // MESH_STRUCT_H
