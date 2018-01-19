#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/constraint_matrix.h>

#include <iostream>

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
    IndexSet                                    mesh_locally_owned;
    IndexSet                                    mesh_locally_relevant;
    ConstraintMatrix                            mesh_constraints;
    ConditionalOStream                        	pcout;

    Mesh_struct<dim>                            mesh_struct;

    void make_grid();


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
    mesh_struct(0.1,0.01),
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
        right_top[0] = 5000; right_top[1] = 300;
        n_cells.push_back(10); n_cells.push_back(3);
    }
    else if (dim == 3){
        right_top[0] = 5000; right_top[1] = 5000; right_top[2] = 300;
        n_cells.push_back(10); n_cells.push_back(10); n_cells.push_back(3);
    }

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      n_cells,
                                                      left_bottom,
                                                      right_top,
                                                      true);
}

template <int dim>
void mm_test<dim>::run(){
    // after we generated the mesh we update the our custom Mesh structure
    mesh_struct.updateMeshStruct(mesh_dof_handler,
                                 mesh_fe,
                                 mesh_constraints,
                                 mesh_locally_owned,
                                 mesh_locally_relevant,
                                 mesh_vertices,
                                 distributed_mesh_vertices,
                                 mpi_communicator,
                                 pcout);
}

int main (int argc, char **argv){
    deallog.depth_console (1);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    mm_test<2> mm;
    mm.run();

    return 0;
}
