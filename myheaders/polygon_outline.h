#ifndef POLYGON_OUTLINE_H
#define POLYGON_OUTLINE_H

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/assign.hpp>

#include <deal.II/base/point.h>

typedef boost::geometry::model::d2::point_xy<double> b_point;
typedef boost::geometry::model::polygon<b_point> b_polygon;


template<int dim>
class Polygon_Outline{
public:
    Polygon_Outline();
    std::vector<b_polygon> Polygons;

    void addPolygon(std::vector<dealii::Point<dim>> newPoly);

    bool isPointInside(dealii::Point<dim> p );

    void PrintPolygons(unsigned int rank);

    void condense();

};

template <int dim>
Polygon_Outline<dim>::Polygon_Outline(){
    if (dim != 2){
        std::cerr << "The Polygon_Outline class cannot be used in other than 2D" << std::endl;
    }
    Polygons.clear();
}

template <int dim>
void Polygon_Outline<dim>::addPolygon(std::vector<dealii::Point<dim> > newPoly){
    if (dim != 2){
        std::cerr << "The Polygon_Outline class cannot be used in other than 2D" << std::endl;
        std::cerr << "No polygon is added" << std::endl;
        return;
    }
    b_polygon b_newpoly;
    std::vector<b_point> pnts;

    for (unsigned int i = 0; i < newPoly.size(); ++i){
        pnts.push_back(b_point(newPoly[i][0],newPoly[i][1]));
    }
    boost::geometry::assign_points(b_newpoly, pnts);
    boost::geometry::correct(b_newpoly);

    if (Polygons.size() == 0){
        Polygons.push_back(b_newpoly);
    }else{
        std::vector<b_polygon> unionRes;
        bool is_new = true;
        for (unsigned int i = 0; i < Polygons.size(); ++i){
            boost::geometry::union_(Polygons[i], b_newpoly, unionRes);
            if (unionRes.size() == 1){
                b_polygon simplified;
                boost::geometry::simplify(unionRes[0], simplified, 0.01);
                Polygons.at(i) = simplified;
                is_new = false;
                break;
            }
        }

        if (is_new){
            Polygons.push_back(b_newpoly);
        }

        //condense();
    }
}

template <int dim>
bool Polygon_Outline<dim>::isPointInside(dealii::Point<dim> p){
    bool in = false;
    for (unsigned int i = 0; i < Polygons.size(); ++i){
        in = boost::geometry::covered_by(b_point(p[0],p[1]),Polygons[i]);
        if (in)
            break;
    }
    return in;
}

template <int dim>
void Polygon_Outline<dim>::PrintPolygons(unsigned int rank){
    std::cout << "Rank " << rank << " has " << Polygons.size() << " polygons" << std::endl;
    for (unsigned int i = 0; i < Polygons.size(); ++i){
        std::cout << "Rank " << rank << " : " << boost::geometry::dsv(Polygons[i]) << std::endl;
    }
}

template<int dim>
void Polygon_Outline<dim>::condense(){
    std::vector<int> dlt;
    for (unsigned int i = 0; i < Polygons.size(); ++i){
        for (unsigned int j = 0; j < Polygons.size(); ++j){
            if (i == j)
                continue;
            std::vector<b_polygon> unionRes;
            boost::geometry::union_(Polygons[i], Polygons[j], unionRes);
            if (unionRes.size() == 1){
                b_polygon simplified;
                boost::geometry::simplify(unionRes[0], simplified, 0.01);
                Polygons.at(i) = simplified;
                dlt.push_back(j);
            }
        }
    }

    std::vector<b_polygon> temp;
    temp = Polygons;
    std::cout << temp.size() << std::endl;
    Polygons.clear();
    for (unsigned int i = 0; i < temp.size(); ++i){
        bool dlt_this = false;
        for (int j = 0; j < dlt.size(); ++j){
            if (i == dlt[j]){
                dlt_this = true;
                break;
            }
        }
        if (!dlt_this){
            Polygons.push_back(temp[i]);
        }
    }
}




#endif // POLYGON_OUTLINE_H
