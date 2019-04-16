from sympy import *
"""
Assembly of BDM
---------------

* loop cells
  - loop quadrature
    + pullback (A)
    + loop local basis
      * loop local basis
	- assemble (B)
"""
N,d,A,B,C,D,E = symbols('N d A B C D E')
Nc = N**d       # total number of hex cells
Nq = 2**d       # number of quadrature points per cell
Nl = 1 + d*2**d # number of local basis functions
Cbdm = Nc*Nq*(A+Nl**2*B)

"""
Assembly of WY
--------------

* loop cells
  - loop quadrature
    + pullback (A)
    + loop dim
      * loop dim
	- assemble (C)

* loop vertices v
  - loop cells(v)
    + loop dim
      * loop dim
        - assemble (D)
  - stencil (E)
"""
Nc = N**d       # total number of hex cells
Nq = 2**d       # number of quadrature points per cell
Nv = Nc         # about the same number of vertices as cells
Ns = 2**d       # how many cells touch a single vertex (=Nq)
Cwy = Nc*Nq*(A+d**2*C) + Nv*(Ns*d**2*D+E)
Cwy = Nc*Nq*(A+d**2*C) + Nc*(Nq*d**2*D+E)   # Nv = Nc, Ns = Nq
Cwy = Nc*Nq*(A+d**2*C) + Nc*Nq*d**2*D+Nc*E
Cwy = Nc*Nq*( (A+d**2*C) + d**2*D + E/Nq)

R = Cwy / Cbdm
# (A + C*d**2 + D*d**2 + 2**(-d)*E)/(A + B*(2**d*d + 1)**2)
print(R.subs({d:3}))
# (A + 9*C + 9*D + E/8)/(A + 625*B)
