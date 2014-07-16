function outputs=run_benchmark(imnames, topchosen, topscores, sptextdir, regspimgdir, sbddir, categids, refineddir)

%first compute all overlaps
overlaps=cell(numel(imnames),1);
for i=1:numel(imnames)
    if(rem(i-1,10)==0) fprintf('Computing overlaps:%d/%d\n',i, numel(imnames)); end
    %read the sprep
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
    
    %load gt
    [cls, inst, categories]=load_gt(sbddir, imnames{i});

    %for each category of interest
    for j=1:numel(categids)
        categ_reg2sp=reg2sp(:,topchosen{categids(j)}{i}); 
   


        %if there is refined regions, read them
        if(exist('refineddir', 'var') & ~isempty(topchosen{categids(j)}{i}))
            tmp=load(fullfile(refineddir, int2str(categids(j)), [imnames{i} '.mat']));
            categ_reg2sp=tmp.newreg2sp;
        end
    
        %compute overlaps
        [overlap, pprecision, precall]=get_gt_overlaps(logical(categ_reg2sp), sp, double(inst));

        overlaps{j}{i}=overlap;
    end    
    gt{i}=categories;

    
end


%now run the evaluation. This is relatively fast once overlaps are precomputed
ap_vol=zeros(9,numel(categids));
for j=1:numel(categids)
    for t=1:9
        outputs(t,j)=generalized_det_eval(imnames, topscores{categids(j)}, overlaps{j}, gt, categids(j), 0.1, 0.1*t);
        ap_vol(t,j)=outputs(t,j).PR.ap;
        fprintf('Evaluated threshold:%f for category:%d\n', 0.1*t, categids(j));
    end
end

%Print out ap_vol for all categories
categ_names_and_groups;
fprintf('Category name \t\t | AP^r\t | AP^r_{vol}\n');
fprintf('_______________________________________________________\n');
for j=1:numel(categids)
    fprintf('%s \t\t | %f\t | %f\n',categnames{categids(j)}, ap_vol(5,j), mean(ap_vol(:,j)));
end
fprintf('_______________________________________________________\n');
fprintf('Mean \t\t | %f\t | %f\n', mean(ap_vol(5,:)), mean(mean(ap_vol,2)));




%we will only work with the 0.5 threshold now
%produce the impact chart
produce_impact_chart_all(outputs(5,:));
output.ap_vol=ap_vol;
 
