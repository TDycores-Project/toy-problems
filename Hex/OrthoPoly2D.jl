function poly2d(px, py)
    """
    Construct a 2D polynomial based on 1D basis in x and y directions.
    Input:
    px: polynomial in x direction or direction 1
    py: polynomial in y direction or direction 2
    Return:
    Matrix p, which is the coefficients of a 2D polynomial p(x,y)
    """
    px = reshape(px, :)
    py = reshape(py, :)
    # degree of poly in x
    n = length(px)
    # degree of poly in y
    m = length(py)
    # construct poly coefficients in 2D and store it in p
    p = zeros(n, m)
    for i=1:length(px)
        for j=1:length(py)
            p[i,j] = px[i] * py[j]
        end
    end
    
    return p
end

function polystring2d(p, var1, var2; threshold=1e-12)
    """Construct a string representation of the polynomial p(x,y) in 2D
    p: polynomial coefficients in x and y directions created by poly2d function
    var1: the 1st variable of polynomial for displaying, can be "x", "y", or "z"
    var2: the 2nd variable of polynomial for displaying, can be "x", "y", or "z"
    """
    n,m = size(p)
    k = n * m
    pol = Array{String}(undef, k)
    kk = 1
    for j=1:m
        for i=1:n
            a = p[i,j] 
            if (abs(a) < threshold) 
                a = 0.0
            end
            pol[kk] = "$a $var1^$(i-1) $var2^$(j-1)"
            kk += 1 
        end
    end
            
    join((pol[i] for i=1:k), " + ")
end

function polydisp2d(name, var1, var2, p)
    """Display the polynomial with friendly math formatting.
    p: polynomial coefficients in x and y directions created by poly2d function
    """
    
    s = join(('$', "$name($var1,$var2) = ", polystring2d(p, var1, var2), '$'))
    # If you download this notebook, you can uncomment the following
    # line to display rendered math.
    display("text/latex", s)
    # This feature is broken in Ed's "Saturn" so we'll just display
    # plain text.
    #display("text/plain", s)
end

function polyval2d(p, x, y)
    """Evaluate function described by coefficients p at point (x,y).
    Input:
    p: polynomial coefficients in 2D build from poly2D(px,py) function 
    Return:
    The value of p(x,y) at (x,y). Note (x,y) can be provided as a vector
    """
    # first we compute each column using Horner's rule
    # then multiply by y^(j-1), we can represent matrix P by
    # 1st col=y^0 * [f1_n(x)], 2nd col=y^1 * [f2_n(x)],... OR
    # 1st row=x^0 * [g1_m(y)], 2nd row=x^1 * [g2_m(y)],... 
    # we use the first one to evaluate poly in 2D
    n,m = size(p)
    g = zero.(x)
    for j = 1:m
        p1 = zeros(n)
        p1[:] = p[:,j]
        f = one.(x) * p1[end]
        for a in reverse(p1[1:end-1])
            f = a .+ x .* f
        end
        g += f .* y.^(j-1)
    end
    return g
end

function polyderiv2d(p, deriv)
    """Evaluate dp/d1 or dp/d2 which means derivative of p with respect to 1st and 2nd variable
    Input:
    p: polynomial coefficients in 2D 
    deriv: can be "d1" or "d2"
    Return:
    dp/d1 or dp/d2 as a 2D array
    """
    # we can represent matrix P by
    # 1st col=y^0 * [f1_n(x)], 2nd col=y^1 * [f2_n(x)],... 
    n,m = size(p)
    n1 = n-1
    m1 = m-1
    if deriv =="d2" && m1 > 0.
        dp = zeros(n,m1)
        for j = 1:m1
            dp[:,j] = p[:,j+1] * j
        end
    elseif deriv == "d1" && n1 > 0.
        dp = zeros(n1,m)
        for i = 1:n1
            dp[i,:] = p[i+1,:] * i
        end
    elseif deriv =="d2" && m1 == 0
        dp = zeros(1,1)
    elseif deriv == "d1" && n1 == 0.
        dp = zeros(1,1)
    else
        error("""deriv should be "d1" or "d2" """)
    end
    return dp
end

function polyaddsub1d(p, q, mode)
    """
    Addition/subtraction of two 1D polynomials p and q.
    (maybe we don't need this function)
    mode: "add" for addition "sub" for subtraction
    """
    p = reshape(p, :)
    q = reshape(q, :)
    n = max(length(p), length(q))
    r = zeros(n)
    r[1:length(p)] = p
    if mode == "add"
        r[1:length(q)] += q
    elseif mode == "sub"
        r[1:length(q)] -= q
    else
        error("mode must be add or sub")
    end
    reshape(r, 1, :)
    
    return r
end

function polyaddsub2d(p, q, mode)
    """
    Add/subtract two polynomials p and q in 2D.
    Input:
    p: Coefficients of 2D polynomial created by poly2D(px,py)
    q: Coefficients of 2D polynomial created by poly2D(qx,qy)
    mode: "add" for addition "sub" for subtraction
    Return:
    P: a matrix which is the coefficients of p(x,y) +/- q(x,y)
    """
    (n1,m1) = size(p)
    (n2,m2) = size(q)
    
    # scatter pp in matrix P1 of size(n,m)
    n = max(n1, n2)
    m = max(m1, m2)
    P1 = zeros(n,m)
    P1[1:n1,1:m1] = p
    # scatter qq in matrix P2 of size(n,m)
    P2 = zeros(n,m)
    P2[1:n2,1:m2] = q
    
    # add or subtract P1 and P2
    if mode == "add"
        P = P1 + P2
    elseif mode == "sub"
        P = P1 - P2
    else
        error("mode must be add or sub")
    end
    
    return P
end

function polymul1d(p, q)
    """Multiply two polynomials in 1D, returinng a polynomial."""
    p = reshape(p, :)
    q = reshape(q, :)
    n = length(p) + length(q) - 1
    r = zeros(n)
    for (i, a) in enumerate(p)
        for (j, b) in enumerate(q)
            r[i+j-1] += a * b
        end
    end
    # return a row vector (for easier viewing)
    reshape(r, 1, :)
    
    return r
end


function polymul2d(p, q)
    """
    Multiply two polynomials p and q in 2D.
    Input:
    p: Coefficients of 2D polynomial created by poly2D(px,py)
    q: Coefficients of 2D polynomial created by poly2D(qx,qy)
    Return:
    P: a matrix which is the coefficients of p(x,y) * q(x,y)
    """
    (n1,m1) = size(p)
    (n2,m2) = size(q)

    k1 = n1+n2-1
    k2 = m1+m2-1
    P1 = zeros(m1,k1,k2)
    for i=1:m1
        p1 = zeros(n1)
        # get column i of p or y^(i-1)f_{n1-1}(x)
        p1 = p[:,i]
        for j=1:m2
            q1 = zeros(n2)
            q1 = q[:,j]
            # multiply column i of p to all columns of q
            r = polymul1d(p1, q1)
            P1[i,:,j+(i-1)] = r
        end
    end
    # add those columns with same degree of y
    P = zeros(k1,k2)
    for i=1:m1
       P[:,:] += P1[i,:,:]
    end

    return P
end

function BilinearMap(Coord_E, xhat, yhat)
    """ 
    This function maps [xhat,yhat] in Ehat=[-1,1]^2 
    to (x,y) in quadrilateral E.
    Input:
    ------
    coord_E: coordinates of quadrilateral E .
    coord_E is 4x2 array
    coord_E = [x1 y1;x2 y2;x3 y3;x4 y4] with vertices numbering
    3----4
    |    |
    1----2
    [xhat, yhat] in Ehat
    Output:
    ------
    x, y: mapped vector in E.
    DF_E: Jacobian matrix
    J_E: det(DF_E)
    """
    m = length(xhat)
    N1 = @. 0.25*(1-xhat)*(1-yhat)
    N2 = @. 0.25*(1+xhat)*(1-yhat)
    N3 = @. 0.25*(1-xhat)*(1+yhat)
    N4 = @. 0.25*(1+xhat)*(1+yhat)
    N = [N1 N2 N3 N4]
    X = N * Coord_E
    # X(2,m), 1st row x, 2nd row y,
    X = X'
    x = X[1,:]
    y = X[2,:]
    # gradient of N, [dN/dxhat; dN/dyhat]
    GradN = zeros(2,m,4)
    GradN[1,:,:] = @. 0.25*[-(1-yhat) (1-yhat) -(1+yhat) (1+yhat)]
    GradN[2,:,:] = @. 0.25*[-(1-xhat) -(1+xhat) (1-xhat) (1+xhat)]

    # JT = [[dx/dxhat, dy/dxhat],
    #       [dx/dyhat, dy/dyhat]] (3m x 3)

    JTxhat = GradN[1,:,:] * Coord_E
    dxdxhat = JTxhat[:,1]
    dydxhat = JTxhat[:,2]
    JTyhat = GradN[2,:,:] * Coord_E
    dxdyhat = JTyhat[:,1]
    dydyhat = JTyhat[:,2]

    # compute det
    detJ = @. dxdxhat*dydyhat - dydxhat*dxdyhat
    
    J = zeros(2,m,2)
    J[1,:,:] = [dxdxhat dxdyhat]
    J[2,:,:] = [dydxhat dydyhat]
    
    return x, y, J, detJ
end

function GetQuadrature2d(Q)
    """
    create Gauss points and weights on [-1, 1]^2
    """
    # 1D quadrature on [-1, 1]
    q = zgj(Q, 0.0, 0.0)
    w = wgj(q, 0.0, 0.0)
    
    W2 = zeros(Q*Q)
    Qx = zeros(Q*Q)
    Qy = zeros(Q*Q)
    for i=1:Q
        for j=1:Q
            k = (i-1)*Q +j
            Qx[k] = q[j]
            Qy[k] = q[i]
            W2[k] = w[j]*w[i]
        end
    end
    return W2, Qx, Qy
end

function poly_inner_product2d(u, v, Coord_E)
    """the L2 inner product of u(x,y) and v(x,y) on quadrilateral E with coord_E
    Both u and v are polynomials expressed as matrix in the usual way.
    """
    P = polymul2d(u, v)
    n,m = size(P)
    # max degrees of poly in x or y
    n = max(n,m)-1
    # No. quadrature points in 1D
    Q = Int(n+1)
    # get weights and quadrature points on [-1,1]^2
    W2, Qx, Qy = GetQuadrature2d(Q)
    # get detJ to updates weights and quadrature points on quadrilateral E
    x, y, J, detJ = BilinearMap(Coord_E, Qx, Qy)
    # evaluvate u*v at quadrature pts on E
    R = polyval2d(P, x, y)
    # loop over quadrature
    inprod_uv = (detJ .* W2)' * R
    
    return inprod_uv 
end


function poly_norm2d(u, Coord_E)
    """Compute the norm of the polynomial u(x,y) induced by poly_inner_product2d().
    """
    inprod_uu = poly_inner_product2d(u, u, Coord_E)

    return sqrt(inprod_uu)
end

function monomials1d(n)
    """Return a list of the first n monomials in order of increasing degree.
    We will think of this as a "tall" matrix indexed by its n columns.
    Each column contains a polynomial.
    For example for n = 3, we have 1, x, x^2
    """
    A = []
    for j in 1:n
        p = zeros(j)
        p[end] = 1
        push!(A, p)
    end
    return A
end


function monomials2d(n, m)
    """
    Input:
    n: number of basis in x direction in order of increasing degree
    m: number of basis in y direction in order of increasing degree
    Return:
    A list which contains matrices for 2D polynomial
    """
    A = monomials1d(n)
    B = monomials1d(m)
    P = []
    for j = 1:m
        q = copy(B[j])
        for i = 1:n
            p = copy(A[i])
            pq = poly2d(p,q)
            push!(P, pq)
        end
    end
    return P
end

function poly_qr2d(n,m, Coord_E)
    """
    n: number of poly basis in x direction
    m: number of poly basis in y direction
    coord_E: coordinates of quadrilateral E
    returns Q, R which Q is a list of size n*m of matrices
    contains orthogonal poly on E
    """
    
    A = monomials2d(n,m)
    Q = []
    R = zeros(n*m, n*m)
    for j = 1:m
        for i = 1:n
            k = i + n*(j-1) # kth column of A
            v = copy(A[k])
            for l = 1:k-1
                r = poly_inner_product2d(Q[l],v, Coord_E)
                R[l, k] = r
                v = polyaddsub2d(v,Q[l]*r,"sub")
            end
            r = poly_norm2d(v, Coord_E)
            R[k,k] = r
            push!(Q,v/r)
        end
    end
            
    return Q, R
end

function poly_QtQ(Q, Coord_E)
    """Compute Q.T Q where Q is a matrix of polynomials
    expressed as a list of matrices. The result is an ordinary
    square matrix.
    """
    n = length(Q)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i,j] = poly_inner_product2d(Q[i], Q[j], Coord_E)
        end
    end
    A
end
