#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/constraint_matrix.h>

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "myheaders/mesh_struct.h"

using namespace dealii;

template <int dim>
class mm_test{
public:
    mm_test();
    ~mm_test();

    void run();

private:

    MPI_Comm                                  	mpi_communicator;
    parallel::distributed::Triangulation<dim> 	triangulation;
    DoFHandler<dim>                             mesh_dof_handler;
    FESystem<dim>                              	mesh_fe;
    TrilinosWrappers::MPI::Vector               mesh_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_vertices;
    TrilinosWrappers::MPI::Vector               mesh_Offset_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_Offset_vertices;
    IndexSet                                    mesh_locally_owned;
    IndexSet                                    mesh_locally_relevant;
    ConstraintMatrix                            mesh_constraints;
    ConditionalOStream                        	pcout;

    Mesh_struct<dim>                            mesh_struct;

    void make_grid();
    void refine_transfer(std::string prefix);
    void refine_transfer1();

    void do_one_random_refinement(double top_fraction, double bottom_fraction);


};

template <int dim>
mm_test<dim>::mm_test()
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::limit_level_difference_at_vertices)),
    mesh_dof_handler (triangulation),
    mesh_fe (FE_Q<dim>(1),dim),
    mesh_struct(0.0001,0.0001),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    make_grid();
}

template <int dim>
mm_test<dim>::~mm_test(){
    mesh_dof_handler.clear();
}

template <int dim>
void mm_test<dim>::make_grid(){
    Point<dim> left_bottom;
    Point<dim> right_top;
    std::vector<unsigned int>	n_cells;
    if (dim == 2) {
        right_top[0] = 10000; right_top[1] = 1000;
        n_cells.push_back(20); n_cells.push_back(5);
    }
    else if (dim == 3){
        right_top[0] = 10000; right_top[1] = 10000; right_top[2] = 1000;
        n_cells.push_back(10); n_cells.push_back(10); n_cells.push_back(3);
    }

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      n_cells,
                                                      left_bottom,
                                                      right_top,
                                                      true);

    // Refine a couple of times so that we start with a more complex mesh to work with
    //for (unsigned int ir = 0; ir < 0; ++ir){
    //    do_one_random_refinement(20, 95);
    //}
}

template <int dim>
void mm_test<dim>::refine_transfer(std::string prefix){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    // first prepare the triangulation
    triangulation.prepare_coarsening_and_refinement();

    //prepare vertices for transfering
    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>mesh_trans(mesh_dof_handler);
    std::vector<const TrilinosWrappers::MPI::Vector *> x_fs_system (1);

    x_fs_system[0] = &mesh_vertices;
    mesh_trans.prepare_for_coarsening_and_refinement(x_fs_system);

    std::cout << "Number of active cells Before: "
                << triangulation.n_active_cells()
                << std::endl;
    std::cout << "dofs 2: " << mesh_dof_handler.n_dofs() << std::endl << std::flush;
    // execute the actual refinement
    triangulation.execute_coarsening_and_refinement ();

    std::cout << "Number of active cells After: "
                << triangulation.n_active_cells()
                << std::endl;
    std::cout << "dofs 3: " << mesh_dof_handler.n_dofs() << std::endl << std::flush;

    //For the mesh
    mesh_dof_handler.distribute_dofs(mesh_fe); // distribute the dofs again
    std::cout << "dofs 4: " << mesh_dof_handler.n_dofs() << std::endl << std::flush;
    mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);

    distributed_mesh_vertices.reinit(mesh_locally_owned, mpi_communicator);
    distributed_mesh_vertices.compress(VectorOperation::insert);

    std::vector<TrilinosWrappers::MPI::Vector *> mesh_tmp (1);
    mesh_tmp[0] = &(distributed_mesh_vertices);

    mesh_trans.interpolate (mesh_tmp);
    mesh_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    mesh_vertices = distributed_mesh_vertices;

    pcout << "moving vertices " << std::endl << std::flush;
    mesh_struct.move_vertices(mesh_dof_handler,
                              mesh_vertices,
                              my_rank, prefix);
}

template <int dim>
void mm_test<dim>::refine_transfer1(){
    std::vector<bool> locally_owned_vertices = triangulation.get_used_vertices();
    {
        // Create the boolean input of communicate_locally_moved_vertices method
        // see implementation of GridTools::get_locally_owned_vertices in grid_tools.cc line 2172 (8.5.0)
        typename parallel::distributed::Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell != endc; ++cell){
            if (cell->is_artificial() ||
                    (cell->is_ghost() && cell->subdomain_id() < triangulation.locally_owned_subdomain() )){
                for (unsigned int v = 0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    locally_owned_vertices[cell->vertex_index(v)] = false;
            }
        }
    }

    // Call the method before
    triangulation.communicate_locally_moved_vertices(locally_owned_vertices);

    {// Apply the opposite displacement
        std::map<types::global_dof_index, bool> set_dof;
        std::map<types::global_dof_index, bool>::iterator it_set;
        typename DoFHandler<dim>::active_cell_iterator
        cell = mesh_dof_handler.begin_active(),
        endc = mesh_dof_handler.end();
        for (; cell != endc; ++cell){
            if (cell->is_locally_owned()){
                for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                    Point<dim> &v=cell->vertex(vertex_no);
                    for (unsigned int dir=0; dir < dim; ++dir){
                        types::global_dof_index dof = cell->vertex_dof_index(vertex_no, dir);
                        it_set = set_dof.find(dof);
                        if (it_set == set_dof.end()){
                            v(dir) = v(dir) - mesh_Offset_vertices(dof);
                            set_dof[dof] = true;
                        }
                    }
                }
            }
        }
    }

    triangulation.communicate_locally_moved_vertices(locally_owned_vertices);
    // now the mesh should be consistent as when it was first created
    // so we can hopefully refine it
    triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void mm_test<dim>::run(){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);

    // after we generated the mesh we update the custom Mesh structure
    mesh_struct.updateMeshStruct(mesh_dof_handler,
                                 mesh_fe,
                                 mesh_constraints,
                                 mesh_locally_owned,
                                 mesh_locally_relevant,
                                 mesh_vertices,
                                 distributed_mesh_vertices,
                                 mesh_Offset_vertices,
                                 distributed_mesh_Offset_vertices,
                                 mpi_communicator,
                                 pcout,
                                 "iter0");
    //mesh_struct.printMesh("animBefore_0", my_rank,mesh_dof_handler);



    // Set Top and Bottom elevation
    RBF<dim-1> rbf;
    std::vector<Point<dim-1> > cntrs;
    std::vector<double> wdth;


    if (dim == 2){
        for (unsigned int i = 1; i < 5; ++i){
            Point<dim-1> temp;
            temp[0] = static_cast<double>(i)*2000;
            cntrs.push_back(temp);
            wdth.push_back(0.001);
        }
    }
    else if (dim == 3){
        for (unsigned int i = 1; i < 5; ++i){
            for (unsigned int j = 1; j < 5; ++j){
                Point<dim-1> temp;
                temp[0] = static_cast<double>(i)*1000;
                if (dim == 3)
                    temp[1] = static_cast<double>(j)*1000;
                cntrs.push_back(temp);
                wdth.push_back(0.001);
            }
        }
    }
    rbf.assign_centers(cntrs,wdth);
    rbf.assign_weights(mpi_communicator);




    // Set initial top bottom elebation elevation top to nodes that are local and they lay on the top or the bottom
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = mesh_struct.PointsMap.begin(); it != mesh_struct.PointsMap.end(); ++it){
        double tt = 300 + rbf.eval(it->second.PNT);
        double bb = 0;
        std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
        for (; itz != it->second.Zlist.end(); ++itz){
            if (itz->is_local){
                itz->rel_pos = (itz->z - itz->Bot.z)/(itz->Top.z - itz->Bot.z);
                //if (my_rank == 0){
                //    std::cout << itz->z << " : " << itz->rel_pos << ", (" << itz->Top.z << ", " << itz->Bot.z << ")" << std::endl;
                //}
                if (itz->isTop){
                    itz->z = tt;
                    itz->isZset = true;
                }
                if (itz->isBot){
                    itz->z = bb;
                    itz->isZset = true;
                }
            }
        }
        //it->second.T = 300;// this is supposed to set the initial elevation
        //it->second.B = 0;
        // Here we update the top
        //it->second.T += rbf.eval(it->second.PNT);
        //std::cout << it->second.T << std::endl;
    }

    //return;

    //std::cout << "I'm rank: " << my_rank << " V(20)= " << rbf.eval(20) << std::endl;
    // The structure is used to update the elevation
    mesh_struct.updateMeshElevation(mesh_dof_handler,
                                    triangulation,
                                    mesh_constraints,
                                    mesh_vertices,
                                    distributed_mesh_vertices,
                                    mesh_Offset_vertices,
                                    distributed_mesh_Offset_vertices,
                                    mpi_communicator,
                                    pcout,
                                    "iter0");
    //return;


    mesh_struct.printMesh("animAfter_0", my_rank,mesh_dof_handler);



    // flag cells for refinement
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            int r = rand() % 100 + 1;
            if (r < 20)
                cell->set_refine_flag ();
            else if(r > 95)
                cell->set_coarsen_flag();
        }
    }



    // The refine transfer refines and updates the triangulation and mesh_dof_handler
    //refine_transfer("refine0");
    refine_transfer1();
    mesh_struct.printMesh("animBefore_1", my_rank,mesh_dof_handler);
    return;

    // Then we need to update the custon mesh structure after any change of the triangulation
    mesh_struct.updateMeshStruct(mesh_dof_handler,
                                 mesh_fe,
                                 mesh_constraints,
                                 mesh_locally_owned,
                                 mesh_locally_relevant,
                                 mesh_vertices,
                                 distributed_mesh_vertices,
                                 mesh_Offset_vertices,
                                 distributed_mesh_Offset_vertices,
                                 mpi_communicator,
                                 pcout, "iter1");
    //return;

    //std::cout << "------------------------------------------------------------" << std::endl;

    // modify top function
    for (unsigned int i = 0; i < 5; ++i){
        for (unsigned int j = 0; j < 5; ++j){
            Point<dim-1> temp;
            temp[0] = static_cast<double>(i)*1000 + 500;
            if (dim == 3)
                temp[1] = static_cast<double>(j)*1000 + 500;
            cntrs.push_back(temp);
            wdth.push_back(0.002);
        }
    }
    rbf.assign_centers(cntrs,wdth);
    rbf.assign_weights(mpi_communicator);



    for (it = mesh_struct.PointsMap.begin(); it != mesh_struct.PointsMap.end(); ++it){
        it->second.B = 0;
        it->second.T = 300;
        it->second.T += rbf.eval(it->second.PNT);
        //std::cout << rbf.eval(it->second.PNT) << std::endl;
    }
    //std::cout << "I'm rank: " << my_rank << " V(20)= " << rbf.eval(20) << std::endl;

    mesh_struct.updateMeshElevation(mesh_dof_handler,
                                    triangulation,
                                    mesh_constraints,
                                    mesh_vertices,
                                    distributed_mesh_vertices,
                                    mesh_Offset_vertices,
                                    distributed_mesh_Offset_vertices,
                                    mpi_communicator,
                                    pcout,"iter1");
    mesh_struct.printMesh("animAfter_1", my_rank,mesh_dof_handler);
    //return;


    // ------------------ Second refinment iteration ---------------------------------------------------

    for (int i = 0; i < 10; ++i){
        pcout << "====================== ITER: " << i << "=============================" << std::endl;
        // refine the updated elevations
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell){
            if (cell->is_locally_owned()){
                int r = rand() % 100 + 1;
                if (r < 10)
                    cell->set_refine_flag();
                else if(r > 95)
                    cell->set_coarsen_flag();
            }
        }
        // The refine transfer refines and updates the triangulation and mesh_dof_handler
        refine_transfer("refine" + std::to_string(i+1));
        mesh_struct.printMesh("animBefore_" + std::to_string(i+2) , my_rank, mesh_dof_handler);
        //return;
        if (i == 8)
            return;


        // Then we need to update the custon mesh structure after any change of the triangulation
        mesh_struct.updateMeshStruct(mesh_dof_handler,
                                     mesh_fe,
                                     mesh_constraints,
                                     mesh_locally_owned,
                                     mesh_locally_relevant,
                                     mesh_vertices,
                                     distributed_mesh_vertices,
                                     mesh_Offset_vertices,
                                     distributed_mesh_Offset_vertices,
                                     mpi_communicator,
                                     pcout, "iter" + std::to_string(i+2));



        rbf.assign_weights(mpi_communicator);

        for (it = mesh_struct.PointsMap.begin(); it != mesh_struct.PointsMap.end(); ++it){
            it->second.B = 0;
            it->second.T = 300;
            it->second.T += rbf.eval(it->second.PNT);
        }
        //std::cout << "I'm rank: " << my_rank << " V(20)= " << rbf.eval(20) << std::endl;

        mesh_struct.updateMeshElevation(mesh_dof_handler,
                                        triangulation,
                                        mesh_constraints,
                                        mesh_vertices,
                                        distributed_mesh_vertices,
                                        mesh_Offset_vertices,
                                        distributed_mesh_Offset_vertices,
                                        mpi_communicator,
                                        pcout, "iter" + std::to_string(i+2));

        mesh_struct.printMesh("animAfter_" + std::to_string(i+2) , my_rank, mesh_dof_handler);
    }
}

//! Cells with random value below #refine_perc will be refined, and cells with random value above #coarse_perc will be coarsen
template <int dim>
void mm_test<dim>::do_one_random_refinement(double refine_perc, double coarse_perc){
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            int r = rand() % 100 + 1;
            if (r < refine_perc)
                cell->set_refine_flag ();
            else if(r > coarse_perc)
                cell->set_coarsen_flag();
        }
    }

    // first prepare the triangulation
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();

}

int main (int argc, char **argv){
    deallog.depth_console (1);

    //srand (time(NULL));
    //int rr = time(NULL);
    //std::cout << rr << std::endl;
    //srand(rr);
    //srand(1517505046);
    srand(1522316091);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    //This is going to create a box with uniform bottom at 0 and uniform top 100
    mm_test<2> mm;
    mm.run();


    return 0;
}
