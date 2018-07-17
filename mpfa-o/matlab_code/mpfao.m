%function mpfao()


load mesh_30x30.mat

%mpfaExample2; close all;

nx = sqrt(G.cells.num);
ny = nx;

sub1_v1 = zeros(G.cells.num,2);
sub2_v1 = zeros(G.cells.num,2);
sub3_v1 = zeros(G.cells.num,2);
sub4_v1 = zeros(G.cells.num,2);
sub1_v2 = zeros(G.cells.num,2);
sub2_v2 = zeros(G.cells.num,2);
sub3_v2 = zeros(G.cells.num,2);
sub4_v2 = zeros(G.cells.num,2);

sub1_n1 = zeros(G.cells.num,2);
sub2_n1 = zeros(G.cells.num,2);
sub3_n1 = zeros(G.cells.num,2);
sub4_n1 = zeros(G.cells.num,2);
sub1_n2 = zeros(G.cells.num,2);
sub2_n2 = zeros(G.cells.num,2);
sub3_n2 = zeros(G.cells.num,2);
sub4_n2 = zeros(G.cells.num,2);

sub_G1 = zeros(2,2,G.cells.num);
sub_G2 = zeros(2,2,G.cells.num);
sub_G3 = zeros(2,2,G.cells.num);
sub_G4 = zeros(2,2,G.cells.num);

icell = 0;
for jj = 1:ny
    for ii = 1:nx
        icell = icell + 1;
        
        node_idx = [(jj-1)*(nx+1)+ii (jj-1)*(nx+1)+ii+1 jj*(nx+1)+ii+1 jj*(nx+1)+ii];
        
        node_1 = G.nodes.coords(node_idx(1),:);
        node_2 = G.nodes.coords(node_idx(2),:);
        node_3 = G.nodes.coords(node_idx(3),:);
        node_4 = G.nodes.coords(node_idx(4),:);
        cell_c = G.cells.centroids(icell,:);
        
        face_1 = (node_1 + node_2)/2;
        face_2 = (node_2 + node_3)/2;
        face_3 = (node_3 + node_4)/2;
        face_4 = (node_4 + node_1)/2;
        
        %
        %
        %
        %    |     ^ n^4_2    |        ^ n^3_2 |
        %    |     |          |        |       |
        %  node4 ---------- face3 ---------- node3 ----------
        %    |                |                |
        %    |--->n^4_1       |                |--->n^3_1
        %    |                |                |
        %    |      4         |      3         |
        %    |                |-->v^3_1        |
        %    |                |  =v^4_1        |
        %    |                |                |
        %    |       ^ v^1_2  |       ^ v^3_2  |
        %    |       |=v^4_2  |       |=v^2_2  |
        %    |       |        |       |        |
        %  face4 ------------ c ------------ face2
        %    |                |                |
        %    |                |                |
        %    |                |                |
        %    |                |                |
        %    |      1         |      2         |
        %    |                |-->v^1_1        |
        %    |                |  =v^2_1        |
        %    |--->n^1_1       |                |--->n^2_1
        %    |                |                |
        %    |     ^ n^1_2    |        ^ n^2_2 |
        %    |     |          |        |       |
        %  node1 ---------- face1 ---------- node2
        %
        %
        
        % Compute v vectors
        x1 = face_1; x2 = cell_c; x3 = face_4;
        v_1_1 = compute_right_normal_vector(x1, x2) * norm(x2-x1);
        v_1_2 = compute_right_normal_vector(x2, x3) * norm(x3-x2);
        area_1 = abs(det([x1 1; x2 1; x3 1])/2);
        
        x1 = face_1; x2 = face_2; x3 = cell_c;
        v_2_1 = v_1_1;
        v_2_2 = compute_right_normal_vector(x2, x3) * norm(x3-x2);
        area_2 = abs(det([x1 1; x2 1; x3 1])/2);
        
        x1 = cell_c; x2 = face_2; x3 = face_3;
        v_3_1 = compute_left_normal_vector (x3, x1) * norm(x3-x1);
        v_3_2 = v_2_2;
        area_3 = abs(det([x1 1; x2 1; x3 1])/2);
        
        x1 = cell_c; x2 = face_3; x3 = face_4;
        v_4_1 = v_3_1;
        v_4_2 = v_1_2;
        area_4 = abs(det([x1 1; x2 1; x3 1])/2);
        
        % Compute n vectors
        x1 = node_1; x2 = face_1; x3 = face_4;
        n_1_1 = compute_left_normal_vector(x3, x1);
        n_1_2 = compute_left_normal_vector(x1, x2);
        
        x1 = face_1; x2 = node_2; x3 = face_2;
        n_2_1 = compute_right_normal_vector(x2, x3);
        n_2_2 = compute_left_normal_vector(x1, x2);
        
        x1 = face_2; x2 = node_3; x3 = face_3;
        n_3_1 = compute_right_normal_vector(x1, x2);
        n_3_2 = compute_right_normal_vector(x2, x3);
        
        x1 = face_3; x2 = node_4; x3 = face_4;
        n_4_1 = compute_left_normal_vector(x2, x3);
        n_4_2 = compute_right_normal_vector(x1, x2);
        
        
        sub1_v1(icell,:) = v_1_1;sub1_v2(icell,:) = v_1_2;
        sub2_v1(icell,:) = v_1_1;sub2_v2(icell,:) = v_2_2;
        sub3_v1(icell,:) = v_3_1;sub3_v2(icell,:) = v_3_2;
        sub4_v1(icell,:) = v_4_1;sub4_v2(icell,:) = v_4_2;
        
        sub1_n1(icell,:) = n_1_1;sub1_n2(icell,:) = n_1_2;
        sub2_n1(icell,:) = n_2_1;sub2_n2(icell,:) = n_2_2;
        sub3_n1(icell,:) = n_3_1;sub3_n2(icell,:) = n_3_2;
        sub4_n1(icell,:) = n_4_1;sub4_n2(icell,:) = n_4_2;
        
        T1_1 = norm(face_4 - node_1); T1_2 = norm(face_1 - node_1);
        T2_1 = norm(face_2 - node_2); T2_2 = norm(face_1 - node_2);
        T3_1 = norm(face_2 - node_3); T3_2 = norm(face_3 - node_3);
        T4_1 = norm(face_4 - node_4); T4_2 = norm(face_3 - node_4);
        
        K = diag(rock.perm(icell)*ones(2,1));
        
        sub_G1(:,:,icell) = 1/area_1/2 * ...
            [
            T1_1*n_1_1*K*v_1_1' T1_1*n_1_1*K*v_1_2'
            T1_2*n_1_2*K*v_1_1' T1_2*n_1_2*K*v_1_2'
            ];
        
        sub_G2(:,:,icell) = 1/area_2/2 * ...
            [
            T2_1*n_2_1*K*v_2_1' T2_1*n_2_1*K*v_2_2'
            T2_2*n_2_2*K*v_2_1' T2_2*n_2_2*K*v_2_2'
            ];
        
        sub_G3(:,:,icell) = 1/area_3/2 * ...
            [
            T3_1*n_3_1*K*v_3_1' T3_1*n_3_1*K*v_3_2'
            T3_2*n_3_2*K*v_3_1' T3_2*n_3_2*K*v_3_2'
            ];
        
        sub_G4(:,:,icell) = 1/area_4/2 * ...
            [
            T4_1*n_4_1*K*v_4_1' T4_1*n_4_1*K*v_4_2'
            T4_2*n_4_2*K*v_4_1' T4_2*n_4_2*K*v_4_2'
            ];
    end
end
clear icell

iedge = 0;
for jj = 1:ny-1
    for ii = 1:nx-1
        
        iedge = iedge + 1;
        
        icell_1 = (jj-1)*nx + ii;
        icell_2 = icell_1 + 1;
        icell_3 = icell_1 + nx;
        icell_4 = icell_3 + 1;
        
        G1 = sub_G3(:,:,icell_1);
        G2 = sub_G4(:,:,icell_2);
        G3 = sub_G2(:,:,icell_3);
        G4 = sub_G1(:,:,icell_4);
        
        C1 = [...
            -G1(1,1)   0       -G1(1,2)  0
            0         G4(1,1)  0        G4(1,2)
            0        -G3(2,1)  G3(2,2)  0
            G2(2,1)   0        0       -G2(2,2)
            ];
        
        F1 = [...
            -sum(C1(1,:)) 0             0             0
            0             0             0             -sum(C1(2,:))
            0             0             -sum(C1(3,:)) 0
            0            -sum(C1(4,:))  0             0
            ];
        
        C2 = [...
            G2(1,1)   0        0       -G2(1,2)
            0        -G3(1,1)  G3(1,2)  0
            -G1(2,1)   0       -G1(2,2)  0
            0         G4(2,1)  0        G4(2,2)
            ];
        
        F2 = [...
            0            -sum(C2(1,:))  0             0
            0             0            -sum(C2(2,:))  0
            -sum(C2(3,:))  0             0             0
            0             0             0            -sum(C2(4,:))
            ];
        
        A = C1 - C2;
        B = F2 - F1;
        
        edge_T1(:,:,iedge) = C1 * inv(A)*B  + F1;
        edge_C1(:,:,iedge) = C1;
        edge_C2(:,:,iedge) = C2;
        edge_F1(:,:,iedge) = F1;
        edge_F2(:,:,iedge) = F1;
        
        clear G1 G2 G3 G4 icell_*
    end
    
end

% North BC
jj = ny;
ibc = 0;
for ii = 1:nx-1
    
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_2 = icell_1 + 1;
    
    G1 = sub_G3(:,:,icell_1);
    G2 = sub_G4(:,:,icell_2);
    
    den = G1(1,1) + G2(1,1);
    
    a = (G1(1,1) + G1(1,2))/den;
    b = (G2(1,1) - G2(1,2))/den;
    c = (-G1(1,2)         )/den;
    d = ( G2(1,2)         )/den;
    
    T_in = [
        G1(1,1)+G1(1,2)-G1(1,1)*a -G1(1,1)*b
        G1(2,1)+G1(2,2)-G1(2,1)*a -G1(2,1)*b
        G2(2,1)*a                 -G2(2,1)+G2(2,2)+G2(2,1)*b
        ];
    
    T_bc = [
        -G1(1,2)-G1(1,1)*c -G1(1,1)*d
        -G1(2,2)-G1(2,1)*c -G1(2,1)*d
         G2(2,1)*c         -G2(2,2)+G2(2,1)*d
        ];
    N_Tin(:,:,ibc) = T_in;
    N_Tbc(:,:,ibc) = T_bc;
end
clear G1 G2

% South BC
jj = 1;
ibc = 0;
for ii = 1:nx-1
    
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_2 = icell_1 + 1;
    
    G3 = sub_G2(:,:,icell_1);
    G4 = sub_G1(:,:,icell_2);
    
    den = G3(1,1) + G4(1,1);
    
    a = (G3(1,1) - G3(1,2))/den;
    b = (G4(1,1) + G4(1,2))/den;
    c = ( G3(1,2)         )/den;
    d = (-G4(1,2)         )/den;
    
    T_in = [
        G4(1,1)*a                 -G4(1,1)-G4(1,2)+G4(1,1)*b
        G3(2,1)-G3(2,2)-G3(2,1)*a -G3(2,1)*b
        G4(2,1)*a                 -G4(2,1)-G4(2,2)+G4(2,1)*b
        ];
    
    T_bc = [
        G4(1,1)*c          G4(1,2)+G4(1,1)*d
        G3(2,2)-G3(2,1)*c -G3(2,1)*d
        G4(2,1)*c          G4(2,2)+G4(2,1)*d
        ];
    S_Tin(:,:,ibc) = T_in;
    S_Tbc(:,:,ibc) = T_bc;
end
clear G3 G4


% West BC
ii = 1;
ibc = 0;
for jj = 1:ny-1
    
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_2 = icell_1   + nx;
    
    G2 = sub_G4(:,:,icell_1);
    G4 = sub_G1(:,:,icell_2);
    
    den = G2(2,2) + G4(2,2);
    
    a = (-G2(2,1) + G2(2,2))/den;
    b = ( G4(2,1) + G4(2,2))/den;
    c = ( G2(2,1)         )/den;
    d = (-G4(2,1)         )/den;
    
    T_in = [
        -G2(2,1)+G2(2,2)-G2(2,2)*a -G2(2,2)*b
        G4(1,2)*a                 -G4(1,1)-G4(1,2)+G4(1,2)*b
        -G2(1,1)+G2(1,2)-G2(1,2)*a -G2(1,2)*b
        ];
    
    T_bc = [
        G2(2,1)-G2(2,2)*c -G2(2,2)*d
        G4(1,2)*c          G4(1,1)+G4(1,2)*d
        G2(1,1)-G2(1,2)*c -G2(1,2)*d
        ];
    W_Tin(:,:,ibc) = T_in;
    W_Tbc(:,:,ibc) = T_bc;
end
clear G2 G4


% East BC
ii = nx;
ibc = 0;
for jj = 1:ny-1
    
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_2 = icell_1   + nx;
    
    G1 = sub_G3(:,:,icell_1);
    G3 = sub_G2(:,:,icell_2);
    
    den = G1(2,2) + G3(2,2);
    
    a = ( G1(2,1) + G1(2,2))/den;
    b = (-G3(2,1) + G3(2,2))/den;
    c = (-G1(2,1)          )/den;
    d = ( G3(2,1)          )/den;
    
    T_in = [
        G3(2,2)*a                  G3(2,1)-G3(2,2)+G3(2,2)*b
        G1(1,1)+G1(1,2)-G1(1,2)*a -G1(1,2)*b
        G3(1,2)*a                  G3(1,1)-G3(1,2)+G3(1,2)*b
        ];
    
    T_bc = [
         G3(2,2)*c         -G3(2,1)+G3(2,2)*d
        -G1(1,1)-G1(1,2)*c -G1(1,2)*d
         G3(1,2)*c         -G3(1,1)+G3(1,2)*d
        ];
    E_Tin(:,:,ibc) = T_in;
    E_Tbc(:,:,ibc) = T_bc;
end
clear G1 G3


A = zeros(nx*ny,nx*ny);
b = zeros(nx*ny,1);

iedge = 0;
for jj = 1:ny-1
    for ii = 1:nx-1
        
        iedge = iedge + 1;
        
        icell_1 = (jj-1)*nx + ii;
        icell_2 = icell_1 + 1;
        icell_3 = icell_1 + nx;
        icell_4 = icell_3 + 1;
        
        cols = [icell_1 icell_2 icell_3 icell_4];
        flux_rows_from_to = [
            icell_1 icell_2
            icell_3 icell_4
            icell_1 icell_3
            icell_2 icell_4];
        
        T = edge_T1(:,:,iedge);
        bc_contrib = zeros(size(flux_rows_from_to,1),1);
        
        [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols);
        
        clear T icell_*
    end
end

N_bc_value = 3;
S_bc_value = 4;
W_bc_value = 2;
E_bc_value = 1;

% N_bc_value = 1;
% S_bc_value = 1;
% W_bc_value = 1;
% E_bc_value = 1;

% N_bc_value = 4;
% S_bc_value = 4;
% W_bc_value = 1;
% E_bc_value = 1;


% North BC
jj = ny;
ibc = 0;
for ii = 1:nx-1
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_2 = icell_1 + 1;
    
    T    = N_Tin(:,:,ibc);
    T_bc = N_Tbc(:,:,ibc);
    P_bc = [N_bc_value N_bc_value]'/2;
    bc_contrib = T_bc * P_bc;
    
    cols = [icell_1 icell_2];
    flux_rows_from_to = [
        icell_1 icell_2
        icell_1 0
        icell_2 0];
    
    [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols);
    
    clear T T_bc bc_contrib icell_*;
    
end
%disp(['Error North BC = ' num2str(max(abs(b(872:899)*1e3           -xr1.rhs(872:899))))])


% South BC
jj = 1;
ibc = 0;
for ii = 1:nx-1
    
    ibc = ibc + 1;
    icell_3 = (jj-1)*nx + ii;
    icell_4 = icell_3 + 1;
    
    T    = S_Tin(:,:,ibc);
    T_bc = S_Tbc(:,:,ibc);
    P_bc = [S_bc_value S_bc_value]'/2;
    bc_contrib = T_bc * P_bc;
    
    cols = [icell_3 icell_4];
    flux_rows_from_to = [
        icell_3 icell_4
        0       icell_3
        0       icell_4];
    
    [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols);
    
    clear T T_bc bc_contrib icell_*;
end
%disp(['Error South BC = ' num2str(max(abs(b(2:29)*1e3              -xr1.rhs(2:29))))])

% West BC
ii = 1;
ibc = 0;
for jj = 1:ny-1
    
    ibc = ibc + 1;
    icell_2 = (jj-1)*nx + ii;
    icell_4 = icell_2   + nx;
    
    T    = W_Tin(:,:,ibc);
    T_bc = W_Tbc(:,:,ibc);
    P_bc = [W_bc_value W_bc_value]'/2;
    bc_contrib = T_bc * P_bc;
    
    cols = [icell_2 icell_4];
    flux_rows_from_to = [
        icell_2 icell_4
        0       icell_4
        0       icell_2];
    
    [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols);
    
    clear T T_bc bc_contrib icell_*;
end
%disp(['Error West BC  = ' num2str(max(abs(b([31:30:900-30])*1e3    -xr1.rhs([31:30:900-30]))))])

% East BC
ii = nx;
ibc = 0;
for jj = 1:ny-1
    
    ibc = ibc + 1;
    icell_1 = (jj-1)*nx + ii;
    icell_3 = icell_1   + nx;
    %disp([icell_1 icell_3])
    
    T    = E_Tin(:,:,ibc);
    T_bc = E_Tbc(:,:,ibc);
    P_bc = [E_bc_value E_bc_value]'/2;
    bc_contrib = T_bc * P_bc;
    
    cols = [icell_1 icell_3];
    flux_rows_from_to = [
        icell_1 icell_3
        icell_1 0
        icell_3 0];
    
    [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols);
    
    clear T T_bc bc_contrib icell_*;
end
%disp(['Error East BC  = ' num2str(max(abs(b([60:30:900-30])*1e3 -xr1.rhs([60:30:900-30]))))])

% Southwest corner
icell = 1;
Gmat  = sub_G1(:,:,icell);
P_bc  = [E_bc_value N_bc_value]';
in_contrib = [
    -Gmat(1,1)-Gmat(1,2)
    -Gmat(2,1)-Gmat(2,2)
    ];
bc_contrib = [
    (-Gmat(1,1)-Gmat(1,2))*-P_bc(1)
    (-Gmat(2,1)-Gmat(2,2))*-P_bc(2)
    ];

A(icell,icell) = A(icell,icell) - in_contrib(1);
A(icell,icell) = A(icell,icell) - in_contrib(2);
b(icell)       = b(icell)       + bc_contrib(1);
b(icell)       = b(icell)       + bc_contrib(2);

%disp(['SW === ' num2str(b([1])*1e3-xr1.rhs([1]))])

% Southeast corner
icell = nx;
Gmat  = sub_G2(:,:,icell);
P_bc  = [E_bc_value N_bc_value]';
in_contrib = [
    +Gmat(1,1)-Gmat(1,2)
    +Gmat(2,1)-Gmat(2,2)
    ];
bc_contrib = [
    (+Gmat(1,1)-Gmat(1,2))*-P_bc(1)
    (+Gmat(2,1)-Gmat(2,2))* P_bc(2)
    ];

A(icell,icell) = A(icell,icell) + in_contrib(1);
A(icell,icell) = A(icell,icell) - in_contrib(2);
b(icell)       = b(icell)       - bc_contrib(1); % ???
b(icell)       = b(icell)       - bc_contrib(2);


% northeast corner
icell = nx*ny;
Gmat  = sub_G3(:,:,icell);
P_bc  = [E_bc_value S_bc_value]';
in_contrib = [
    +Gmat(1,1)+Gmat(1,2)
    +Gmat(2,1)+Gmat(2,2)
    ];
bc_contrib = [
    (+Gmat(1,1)+Gmat(1,2))*-P_bc(1)
    (+Gmat(2,1)+Gmat(2,2))*-P_bc(2)
    ];

A(icell,icell) = A(icell,icell) + in_contrib(1);
A(icell,icell) = A(icell,icell) + in_contrib(2);
b(icell)       = b(icell)       - bc_contrib(1);
b(icell)       = b(icell)       - bc_contrib(2);


% northwest corner
icell = nx*ny-nx+1;
Gmat  = sub_G4(:,:,icell);
P_bc  = [W_bc_value S_bc_value]';
in_contrib = [
    -Gmat(1,1)+Gmat(1,2)
    -Gmat(2,1)+Gmat(2,2)
    ];
bc_contrib = [
    (-Gmat(1,1)+Gmat(1,2))*P_bc(1)
    (-Gmat(2,1)+Gmat(2,2))*-P_bc(2)
    ];

A(icell,icell) = A(icell,icell) - in_contrib(1);
A(icell,icell) = A(icell,icell) + in_contrib(2);
b(icell)       = b(icell)       - bc_contrib(1);
b(icell)       = b(icell)       - bc_contrib(2); % ???

%disp(['Error Corners  = ' num2str([b([1 30 900 900-29])*1e3-xr1.rhs([1 30 900 900-29])]') ])

%diff = abs(full(xr1.A(1:nx*ny,1:nx*ny)) -A*1e3);
%disp(['Error Matrix   = ' num2str(max(max(diff))) ])

x = A\b;

% figure;
% plotCellData(G,x,'EdgeColor','black'); colorbar
% colormap jet
% h = colorbar;
% set(gca,'fontweight','bold','fontsize',14)
% caxis([1 4])

%orient landscape
%print -dpdf ~/projects/tdycore/notes/presentations/2018-07-24_SciDAC-PI-Meeting/graffle/mpfo.pdf

figure;
icell = 0;
for jj = 1:ny
    for ii = 1:nx
        icell = icell + 1;
        
        node_idx = [(jj-1)*(nx+1)+ii (jj-1)*(nx+1)+ii+1 jj*(nx+1)+ii+1 jj*(nx+1)+ii];
        
        node_1 = G.nodes.coords(node_idx(1),:);
        node_2 = G.nodes.coords(node_idx(2),:);
        node_3 = G.nodes.coords(node_idx(3),:);
        node_4 = G.nodes.coords(node_idx(4),:);
        
        xy = [node_1; node_2; node_3; node_4; node_1];
        fill(xy(:,1),xy(:,2),x(icell))
        hold all
    end
end
colormap jet
h = colorbar;
set(gca,'fontweight','bold','fontsize',14)
caxis([1 4])

