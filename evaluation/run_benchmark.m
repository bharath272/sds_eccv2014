function output=run_benchmark(imnames, topchosen, topscores, sptextdir, regspimgdir, sbddir, categid, refineddir)

%first compute all overlaps
overlaps=cell(numel(imnames),1);
for i=1:numel(imnames)
    if(rem(i-1,10)==0) fprintf('Computing overlaps:%d/%d\n',i, numel(imnames)); end
    %read the sprep
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
    reg2sp=reg2sp(:,topchosen{i});    


    %if there is refined regions, read them
    if(exist('refineddir', 'var'))
        tmp=load(fullfile(refineddir, [imnames{i} '.mat']));
        reg2sp=tmp.newreg2sp;
    end
    
    %load gt
    [cls, inst, categories]=load_gt(sbddir, imnames{i});

    %compute overlaps
    [overlap, pprecision, precall]=get_gt_overlaps(logical(reg2sp), sp, double(inst));

    overlaps{i}=overlap;
    gt{i}=categories;

    
end


%now run the evaluation. This is relatively fast once overlaps are precomputed
ap_vol=zeros(9,1);
for t=1:9
    outputs(t)=generalized_det_eval(imnames, topscores, overlaps, gt, categid, 0.1, 0.1*t);
    ap_vol(t)=outputs(t).PR.ap;
    fprintf('Evaluated threshold:%f\n', 0.1*t);
end

%we will only work with the 0.5 threshold now
output=outputs(5);
produce_diagnostic_plots(output);
fprintf('AP^r at 0.5: %f\n', ap_vol(5));
fprintf('AP^r_{vol}: %f\n', mean(ap_vol));
output.ap_vol=ap_vol;
 
