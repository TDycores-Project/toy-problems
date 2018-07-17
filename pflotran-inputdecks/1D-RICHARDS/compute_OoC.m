nprob = 6;
nruns = nprob-1;

prob_dirs = {'snes','ts'};
fname = 't00';
time_string = '1.60000E+03 s';

figure;

for iprob = 1:length(prob_dirs)
    prob_dir = prob_dirs{iprob};
    for kk = 1:nprob
        tt = 2^(kk-1);
        dt(kk) = tt/10000;
        filename = sprintf('%s/%s%02d.h5',prob_dir,fname,tt);
        data = h5read(filename,sprintf('/Time:  %s/Liquid_Pressure [Pa]',time_string));
        nx = size(data,1);
        ny = size(data,2);
        nz = size(data,3);
        if sum([nx>1 ny>1 nz>1]) > 1
            error('Logic failed');
        end
        p{kk} = reshape(data,nx*ny*nz,1);
    end
    
    dts = dt(2:end);
    
    for ii = 1:nruns
        e_norm_inf(ii) = norm(p{ii+1} - p{ii},Inf);
    end
    
    tmp = [ones(nruns,1) log(dts)'] \ log(abs(e_norm_inf))';slope_i = tmp(2);
    disp(sprintf('Slope = %0.2f [%s]',slope_i,prob_dir))
    switch iprob
        case 1
            loglog(dts,e_norm_inf,'-sr','linewidth',2,'markersize',12,'markerfacecolor','r');
            text(3e-3,floor(e_norm_inf(1)*1e6)/1e6,sprintf('slope = %0.2f',slope_i),'fontweight','bold','fontsize',20,'color','r')
        case 2
            loglog(dts,e_norm_inf,'-ok','linewidth',2,'markersize',12,'markerfacecolor','k');
            text(3e-3,ceil(e_norm_inf(1)*1e6)/1e6,sprintf('slope = %0.2f',slope_i),'fontweight','bold','fontsize',20)
    end
    hold all
    grid on;
    
    set(gca,'fontweight','bold','fontsize',14)
    xlabel('dt [s]')
    ylabel('L_\infty error [Pa]')
end

