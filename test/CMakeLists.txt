# This file is part of tblite.
# SPDX-Identifier: LGPL-3.0-or-later
#
# tblite is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tblite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tblite.  If not, see <https://www.gnu.org/licenses/>.

if(WITH_API)
  add_subdirectory("api")
endif()
add_subdirectory("unit")
if(TBLITE_WITH_CUSOLVERMP)
  add_executable(tblite-mpi-tester mpi/main.f90)
  target_link_libraries(tblite-mpi-tester PRIVATE tblite-lib mstore::mstore MPI::MPI_Fortran)
  add_test(NAME tblite/cusolvermp COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:tblite-mpi-tester> ${MPIEXEC_POSTFLAGS} cusolvermp)
  set_tests_properties(tblite/cusolvermp PROPERTIES PROCESSORS 2)
endif()
