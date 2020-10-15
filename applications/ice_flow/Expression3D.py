
import dolfin as dl


cpp_code = '''
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
#include <vector>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Constant.h>
class AnisTensor3D : public dolfin::Expression
{
public:
  AnisTensor3D() :
      Expression(3,3),
      theta0(1.),
      theta1(1.),
      alpha(0)
      {
      }
      
    friend class Mollifier;
void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
     double sa = sin(alpha);
     double ca = cos(alpha);
     double c00 = theta0*sa*sa + theta1*ca*ca;
     double c01 = (theta0 - theta1)*sa*ca;
     double c11 = theta0*ca*ca + theta1*sa*sa;
  
     values[0] = c00;
     values[1] = c01;
     values[2] = 0.0;
     values[3] = c01;
     values[4] = c11;
     values[5] = 0.0;
     values[6] = 0.0;
     values[7] = 0.0;
     values[8] = 1.0;
  }
  
  void set(double _theta0, double _theta1, double _alpha)
  {
  theta0 = _theta0;
  theta1 = _theta1;
  alpha  = _alpha;
  }
  
private:
  double theta0;
  double theta1;
  double alpha;
  
};
PYBIND11_MODULE(SIGNATURE, m)
    {
    py::class_<AnisTensor3D, std::shared_ptr<AnisTensor3D>, dolfin::Expression>
    (m, "AnisTensor3D")
    .def(py::init<>())
    .def("set", &AnisTensor3D::set);    
    }
'''

ExpressionModule3D = dl.compile_cpp_code(cpp_code)