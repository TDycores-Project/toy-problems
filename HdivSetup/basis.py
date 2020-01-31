from sympy import *
import numpy as np
from sympy.utilities.autowrap import ufuncify
import matplotlib.pyplot as plt

plot = False
petsc = True

# From Wheeler2009, the BDM1 spaces span
#  a1 x + b1 y + g1 + rx**2 + 2sxy
#  a2 x + b2 y + g2 - 2rxy - sy**2

# spatial coordinates in the reference element x,y \in [0,+1]
x,y = symbols('x y')

# coefficients of the space
a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
vx  = a1*x + b1*y + g1 + r*x**2 + 2*s*x*y
vy  = a2*x + b2*y + g2 - 2*r*x*y - s*y**2

if plot: fig,ax = plt.subplots(nrows=2,ncols=4)
for i in range(4):
    for j in range(2):
        k = 2*i+j
        eqs = [ np.dot([vx,vy],[-1, 0]).subs({x:-1,y:-1}),
                np.dot([vx,vy],[ 0,-1]).subs({x:-1,y:-1}),
                np.dot([vx,vy],[ 1, 0]).subs({x: 1,y:-1}),
                np.dot([vx,vy],[ 0,-1]).subs({x: 1,y:-1}),
                np.dot([vx,vy],[-1, 0]).subs({x:-1,y: 1}),
                np.dot([vx,vy],[ 0, 1]).subs({x:-1,y: 1}),
                np.dot([vx,vy],[ 1, 0]).subs({x: 1,y: 1}),
                np.dot([vx,vy],[ 0, 1]).subs({x: 1,y: 1}) ]
        eqs[k] -= 1
        sol = solve(eqs)
        ux = vx.subs(sol)
        uy = vy.subs(sol)
        
        if petsc:
            def _f(fcn):
                fcn = fcn.replace("x**2","x*x")
                fcn = fcn.replace("y**2","y*y")
                fcn = fcn.replace("x","x[0]")
                fcn = fcn.replace("y","x[1]")
                if "/4" in fcn: fcn = "(%s)*0.25;" % (fcn.replace("/4",""))
                if "/8" in fcn: fcn = "(%s)*0.125;" % (fcn.replace("/8",""))
                return fcn
            print("B[%2d] = " % (2*k)   + _f("%s" % (ux)))
            print("B[%2d] = " % (2*k+1) + _f("%s" % (uy)))
            
        if plot:
            X, Y = np.meshgrid(np.linspace(-1,1,9), np.linspace(-1,1,9))
            uxy = ufuncify((x, y), ux)
            vxy = ufuncify((x, y), uy)
            ax[j,i].quiver(X, Y, uxy(X, Y), vxy(X, Y))
            ax[j,i].set_title("v%d%d = [%s, %s]" % (i+1,j+1,ux,uy))
            ax[j,i].plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],'-k')
            ax[j,i].set_xlim([-1.5,1.5])
            ax[j,i].set_ylim([-1.5,1.5])
            print("b%d%d = [%s, %s]" % (i+1,j+1,ux,uy))
if plot: plt.show()


if False:
    print("PetscReal N11(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v11).evalf()))
    print("PetscReal N12(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v12).evalf()))
    print("PetscReal N21(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v21).evalf()))
    print("PetscReal N22(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v22).evalf()))
    print("PetscReal N31(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v31).evalf()))
    print("PetscReal N32(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v32).evalf()))
    print("PetscReal N41(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v41).evalf()))
    print("PetscReal N42(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v42).evalf()))


