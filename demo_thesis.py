oc__author__ = "Alexandra Diem <alexandra@simula.no>"

import sys
import uuid
import poroelastic as poro
import dolfin as df
import numpy as np

def as_function(f):
    '''For f of type indexed get the corresponding function'''
    # Get the function space of the indexed
    f, index = f.ufl_operands
    V = f.function_space()
    # Get the index as int
    i,  = index.indices()
    i = int(i)
    # The space for the indexed function
    Vi = V.sub(i).collapse()
    fi = df.Function(Vi)
    # Fill it with values
    df.assign(fi, f.sub(i))

    return fi

comm = df.mpi_comm_world()

nx = 50
mesh = df.UnitSquareMesh(nx, nx)
params = poro.ParamParser()
data_dir = str(uuid.uuid4())
params.add_data("Simulation", "dir", data_dir)

print("Simulation ID {}".format(data_dir))
print("Number of cells: {}".format(mesh.num_cells()))

pprob = poro.PoroelasticProblem(mesh, params.p)

# Mark boundary subdomians
class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 1.0) and on_boundary

class Top(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 1.0) and on_boundary

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 0.0) and on_boundary

boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left = Left()
right = Right()
top = Top()
bottom = Bottom()
left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)

# Define Dirichlet boundary conditions
highP = df.Constant(20.0)
zero = df.Constant(0.0)
lowP = df.Constant(2)
pprob.add_solid_dirichlet_condition(zero, boundaries, 1, n=0)
pprob.add_solid_dirichlet_condition(zero, boundaries, 2, n=0)

def set_xdmf_parameters(f):
    f.parameters['flush_output'] = True
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False

# Files for output
N = int(params.p['Parameter']['N'])
f1 = [df.XDMFFile(comm, '../data/{}/uf{}.xdmf'.format(data_dir, i)) for i in range(N)]
f2 = [df.XDMFFile(comm, '../data/{}/mf{}.xdmf'.format(data_dir, i)) for i in range(N)]
f3 = [df.XDMFFile(comm, '../data/{}/p{}.xdmf'.format(data_dir, i)) for i in range(N)]
f4 = df.XDMFFile(comm, '../data/{}/du.xdmf'.format(data_dir))

for f in (f1, f2, f3):
    [set_xdmf_parameters(f[i]) for i in range(N)]
set_xdmf_parameters(f4)

dx = df.Measure("dx")
ds = df.Measure("ds")(subdomain_data=boundaries)

sum_fluid_mass = 0
theor_fluid_mass = 0
sum_disp = 0
domain_area = 1.0
phi = params.p['Parameter']["phi"]
rho = params.p['Parameter']["rho"]
qi = params.p['Parameter']["qi"]
dt = params.p['Parameter']["dt"]
tf = params.p['Parameter']["tf"]
avg_error = []
for Mf, Uf, p, Us, t in pprob.solve():

    dU, L = Us.split(True)

    [poro.write_file(f1[i], Uf[i], 'uf{}'.format(i), t) for i in range(N)]
    [poro.write_file(f2[i], as_function(Mf[i]), 'mf{}'.format(i), t) for i in range(N)]
    [poro.write_file(f3[i], p[i], 'p{}'.format(i), t) for i in range(N)]
    poro.write_file(f4, dU, 'du', t)

    #domain_area += df.assemble(df.div(dU)*dx)*(1-phi)
    #sum_fluid_mass += df.assemble(Mf*dx)
    #theor_fluid_mass += qi*rho*dt
    #theor_sol = theor_fluid_mass*domain_area
    #sum_disp += df.assemble(dU[0]*ds(4))
    #avg_error.append(np.sqrt(((df.assemble(Mf*dx)-theor_sol)/theor_sol)**2))
    #print(theor_sol, df.assemble(Mf*dx))

for f in (f1, f2, f3):
    [f[i].close() for i in range(N)]
f4.close()
#error = sum(avg_error)/len(avg_error)

params.write_config('../data/{}/{}.cfg'.format(data_dir, data_dir))
#print("Expected sum fluid mass: {}".format(theor_fluid_mass))
#print("Sum fluid mass: {}".format(sum_fluid_mass))
#print("Average error over all time steps: {}".format(error))
