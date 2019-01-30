from sympy import *

# From Wheeler2009, the BDM1 spaces span
#  a1 x + b1 y + g1 + rx**2 + 2sxy
#  a2 x + b2 y + g2 - 2rxy - sy**2

# spatial coordinates in the reference element x,y \in [-1,+1]
x,y = symbols('x y')

# coefficients of the space
a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')

# degrees of freedom for basis functions [vertex,direction]
v11,v12,v21,v22,v31,v32,v41,v42 = symbols('v11 v12 v21 v22 v31 v32 v41 v42')

# where along the edge do you want the dofs, psi is lower end of
# [-1,1], eta chosen to be symmetric
psi = -1
eta = -psi
vx  = a1*x + b1*y + g1 + r*x**2 + 2*s*x*y
vy  = a2*x + b2*y + g2 - 2*r*x*y - s*y**2
eqs = (v11 - -vx.subs({x: -1,y:psi}),
       v12 - -vy.subs({x:psi,y: -1}),
       v21 - +vx.subs({x: +1,y:psi}),
       v22 - -vy.subs({x:eta,y: -1}),
       v31 - -vx.subs({x: -1,y:eta}),
       v32 - +vy.subs({x:psi,y: +1}),
       v41 - +vx.subs({x: +1,y:eta}),
       v42 - +vy.subs({x:eta,y: +1}))
sol = solve(eqs)
vx  = collect(expand(vx.subs(sol)),[v11,v12,v21,v22,v31,v32,v41,v42])
vy  = collect(expand(vy.subs(sol)),[v11,v12,v21,v22,v31,v32,v41,v42])

if False:
    print("import numpy as np")
    print("import matplotlib.pyplot as plt")
    print("X,Y = np.meshgrid(np.linspace(-1,1,11),np.linspace(-1,1,11))")
    print("def N11(x,y): return %s" % (vx.coeff(v11).evalf()))
    print("def N12(x,y): return %s" % (vy.coeff(v12).evalf()))
    print("def N21(x,y): return %s" % (vx.coeff(v21).evalf()))
    print("def N22(x,y): return %s" % (vy.coeff(v22).evalf()))
    print("def N31(x,y): return %s" % (vx.coeff(v31).evalf()))
    print("def N32(x,y): return %s" % (vy.coeff(v32).evalf()))
    print("def N41(x,y): return %s" % (vx.coeff(v41).evalf()))
    print("def N42(x,y): return %s" % (vy.coeff(v42).evalf()))
    print("plt.quiver(X,Y,N11(X,Y),N12(X,Y))")
    print("plt.xlim(-1.5,+1.5)")
    print("plt.ylim(-1.5,+1.5)")
    print("plt.show()")
    
if True:
    print("PetscReal N11(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v11).evalf()))
    print("PetscReal N12(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v12).evalf()))
    print("PetscReal N21(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v21).evalf()))
    print("PetscReal N22(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v22).evalf()))
    print("PetscReal N31(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v31).evalf()))
    print("PetscReal N32(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v32).evalf()))
    print("PetscReal N41(PetscReal x,PetscReal y) return %s; " % (vx.coeff(v41).evalf()))
    print("PetscReal N42(PetscReal x,PetscReal y) return %s; " % (vy.coeff(v42).evalf()))


