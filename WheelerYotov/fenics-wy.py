from dolfin import *
import matplotlib.pyplot as plt

# Define mesh
N    = 64
mesh = UnitSquareMesh(N,N)
n    = FacetNormal(mesh)

# Define finite elements spaces and build mixed space
BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG" , mesh.ufl_cell(), 0)
V   = FunctionSpace(mesh , BDM * DG)

# Define trial and test functions
(u, p) = TrialFunctions(V)
(v, w) = TestFunctions(V)

# Method of manufactured solutions
g = Expression("pow(1-x[0],4)+pow(1-x[1],3)*(1-x[0])+sin(1-x[1])*cos(1-x[0])", degree=2)
f = Expression("(-5*(12*pow(-x[0] + 1,2) + sin(x[1]-1)*cos(x[0]-1))-1*(3*pow(-x[1] + 1,2) + sin(x[0]-1)*cos(x[1]-1))-1*(3*pow(-x[1] + 1,2) + sin(x[0]-1)*cos(x[1]-1))-2*(-3*(-x[0] + 1)*(2*x[1]-2) + sin(x[1]-1)*cos(x[0]-1)))", degree=2)

K11  = Constant( 2/9.)
K12  = Constant(-1/9.)
K22  = Constant( 5/9.)
Kinv = as_matrix((((K11,K12),(K12,K22))))
              
# Define variational form
a = (dot(Kinv*u, v) - p*div(v) - div(u)*w)*dx
L = -f*w*dx - g*inner(v,n)*ds

# Compute solution
sol = Function(V)
solve(a == L, sol)
(u, p) = sol.split()

# Plot sigma and u
plt.figure()
ax = plot(u,cmap='jet')
plt.colorbar(ax)
plt.savefig("u.png")
plt.close()

plt.figure()
ax = plot(p,cmap='jet')
plt.colorbar(ax)
plt.savefig("p.png")
plt.close()
