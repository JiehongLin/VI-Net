
#include "sphericalmap.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sphericalmap", &sphericalmap);
}
