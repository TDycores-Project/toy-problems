using ArgParse

include("PrismSolverRefElem.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
    	"--N"
            help = "nx=ny=nz=N, number of element in one direction"
            arg_type = Int64
            default = 5
            
        "--a"
            help = "length of the domain in x: [0,a]"
            arg_type = Float64
            default = 1.
            
        "--b"
            help = "length of the domain in y: [0,b]"
            arg_type = Float64
            default = 1.
            
        "--c"
            help = "length of the domain in z: [0,c]"
            arg_type = Float64
            default = 1.
   
   	"--Q1d"
            help = "number of quadrature points in 1D"
            arg_type = Int64
            default = 5
        "--Q_tri"
            help = "number of quadrature in ref triangle"
            arg_type = Int64
            default = 25
            
        "--mesh"
            help = "type of mesh distribution"
            arg_type = String
            default = "uniform"

        "--MMS"
            help = "type of MMS solution"
            arg_type = String
            default = "quartic"

        "--problem"
            help = "type of problem, convergence rate or InfSup test"
            arg_type = String
            default = "convrate"
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
N = parsed_args["N"]
a = parsed_args["a"]
b = parsed_args["b"]
c = parsed_args["c"]
Q1d = parsed_args["Q1d"]
Q_tri = parsed_args["Q_tri"]
mesh = parsed_args["mesh"]
MMS = parsed_args["MMS"]
problem = parsed_args["problem"]

if MMS == "quartic"
    uexact(x,y,z) = VelocityQuartic(x, y, z, 1, 1, 1)
    pexact(x,y,z) = PressureQuartic(x, y, z, 1, 1, 1)
    Forcing(x,y,z) = ForcingQuartic(x, y, z, 1, 1, 1)
elseif MMS == "trig"
    uexact(x,y,z) = VelocityTrig(x, y, z)
    pexact(x,y,z) = PressureTrig(x, y, z)
    Forcing(x,y,z) = ForcingTrig(x, y, z)
else
    println("MMS can be quartic or trig")
end

theta_z = 30
# mesh can be "uniform", "nonuniform", "terrain" "random"
n = 4
p = PlotMesh(mesh, n, n, n, a, b, c, theta_z)
plot(p)
xlabel!("x")
ylabel!("y")
name1 = "$mesh-mesh-$c.png"
savefig(name1)

# we run convergence/infsup from number of element nx=ny=nz=2 up to nx=ny=nz=N

if problem == "convrate"

    eu = zeros(N-1)
    ep = zeros(N-1)
    H = zeros(N-1)

    for i=1:N-1

    	nx = i+1
    	ny = i+1
    	nz = i+1

    	Coord_Ns = GetCoordNodes(mesh,nx, ny, nz, a, b, c, theta_z)
    	IENf, IENn = GetConnectivity(nx, ny, nz)
    	ID = GetID(nx, ny, nz)
    	LM = GetLM(ID, IENf)
    	M, B, F1 = Assembly(Coord_Ns, LM, IENn, Q1d, Q_tri, Forcing)
    	T = GetGlobalTraction(nx, ny, nz, Coord_Ns, Q1d, Q_tri, "all", pexact)
    	F = F1 - T
    	uh, ph = GetFESol(M, B, F)
    
    	ue = GetExactU(Coord_Ns, LM, IENn, uexact)
    	pe = GetExactP(Coord_Ns, IENn, pexact)
    
    	eu[i] = norm(ue - uh) / norm(ue)
    	ep[i] = norm(pe - ph) / norm(pe)
    
    	h = Gethsize(Coord_Ns, IENn)
    	H[i] = h
    end
    H2u = eu[1]*(H/H[1]).^2
    H2p = ep[1]*(H/H[1]).^1
    plot(H, eu, xaxis=:log, yaxis=:log, lw = 3, linestyles = :solid, color="black", label = "Velocity")
    plot!(H, ep, xaxis=:log, yaxis=:log, lw = 3, linestyles = :solid, color= "blue", label = "Pressure")
    plot!(H, H2u, xaxis=:log, yaxis=:log, lw = 2, linestyles = :dash, color="red", label = "O(h^2)")
    plot!(H, H2p, xaxis=:log, yaxis=:log, lw = 2, linestyles = :dash, color="green",label="O(h^1)" ,legend=:bottomright)
    xlabel!("h")
    ylabel!("relative errors")
    name2 = "$problem-$mesh-$c-Ref.png"
    savefig(name2)

end


if problem == "infsup"

    beta = zeros(N-1)
    alph = zeros(N-1)
    H = zeros(N-1)

    for i=1:N-1

    	nx = i+1
    	ny = i+1
    	nz = i+1

    	Coord_Ns = GetCoordNodes(mesh,nx, ny, nz, a, b, c, theta_z)
    	IENf, IENn = GetConnectivity(nx, ny, nz)
    	ID = GetID(nx, ny, nz)
    	LM = GetLM(ID, IENf)
  	M, B, F1 = Assembly(Coord_Ns, LM, IENn, Q1d, Q_tri, Forcing)
    	S, B, C = GetGlobalMat(Coord_Ns, LM, IENn, Q1d, Q_tri)
    	aa, bb = GetInfSupConst(M, S, B, C)
    	beta[i] = bb
    	alph[i] = aa
    
    	h = Gethsize(Coord_Ns, IENn)
    	H[i] = h
    end
    plot(H, beta, xaxis=:log, lw = 3,ylims=Plots.widen(0, 1), legend=false)
    xlabel!("h")
    ylabel!("Inf-Sup constant")
    name3 = "infsup-$mesh-$c-Ref.png"
    savefig(name3)
    plot(H, alph, xaxis=:log, lw = 3,ylims=Plots.widen(0, 1), legend=false)
    xlabel!("h")
    ylabel!("Coercivity constant")
    name3 = "coercivity-$mesh-$c-Ref.png"
    savefig(name3)
end


if problem == "normaltrace"

    mt = zeros(N-1)
    H = zeros(N-1)

    for i=1:N-1

    	nx = i+1
    	ny = i+1
    	nz = i+1

    	Coord_Ns = GetCoordNodes(mesh,nx, ny, nz, a, b, c, theta_z)
    	IENf, IENn = GetConnectivity(nx, ny, nz)
    	
    	max_trace = GetGlobalMaxNormalTrace(Coord_Ns, IENn, Q1d, Q_tri)
    	mt[i] = max_trace
    
    	h = Gethsize(Coord_Ns, IENn)
    	H[i] = h
    end
    plot(H, mt, xaxis=:log, yaxis=:log, lw = 3, linestyles = :solid, color="black", label = false)
    xlabel!("h")
    ylabel!("maximum of normal trace")
    name4 = "$problem-$mesh-$c-Ref.png"
    savefig(name4)
end
