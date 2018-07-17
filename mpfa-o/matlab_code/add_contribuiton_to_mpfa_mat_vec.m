function [A,b] = add_contribuiton_to_mpfa_mat_vec(A,b,T,bc_contrib,flux_rows_from_to,cols)

for kk = 1:size(flux_rows_from_to,1)
    for mm = 1:length(cols)
        col = cols(mm);
        row = flux_rows_from_to(kk,1);
        if (row>0);
            A(row,col) = A(row,col) + T(kk,mm);
            b(row,1  ) = b(row,1)   - bc_contrib(kk);
        end
        
        row = flux_rows_from_to(kk,2);
        if( row>0);
            A(row,col) = A(row,col) - T(kk,mm);
            b(row,1  ) = b(row,1)   + bc_contrib(kk);
        end
    end
end
