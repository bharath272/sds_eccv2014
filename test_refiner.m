function [refined_overlaps, refined_pprecision, refined_precall]=test_refiner(imnames, chosenregs, region_meta_info, sbddir, featdir, sptextdir, regspimgdir, refinement_model, Wsz)
refined_overlaps=cell(numel(imnames),1);
refined_pprecision=cell(numel(imnames),1);
refined_pprecall=cell(numel(imnames),1);
for i=1:numel(imnames)
    tic;
    if(isempty(chosenregs{i})) continue; end
    regids=chosenregs{i};

    %the index into features requires one to take gt into ac
    nongt=find(region_meta_info.gt{i}==0);
    regids_withgt=nongt(regids);

    %load features
    d=load(fullfile(featdir, [imnames{i} '.mat']));
    f=d.feats(regids_withgt,:)';

    %load superpixels
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
       
    %get boxes
    boxes=get_region_boxes(sp, reg2sp);
    spf=[];
    %for each selected region
    for j=1:numel(regids)
        box=boxes(regids(j),:);
        box=expand_box(box, 16/(227-32));
        msk=reg2sp(:,regids(j));
        msk=msk(sp);
        clipped_msk=clip_img(msk, box);
        cl_msk_rsz=imresize(clipped_msk,Wsz);
        spf=[spf cl_msk_rsz(:)];
    end
    
    f=[f; spf];
    %make all predictions
    scr=bsxfun(@plus, refinement_model.W(:,1:end-1)*f, refinement_model.W(:,end));
    pred=1./(1+exp(-scr));    

    newreg2sp=zeros(size(reg2sp,1), numel(regids));
    %for each selected region
    for j=1:numel(regids)
        box=boxes(regids(j),:);
        box=expand_box(box, 16/(227-32));
        orig_reg2sp=reg2sp(:,regids(j));
        coarse_reg2sp=apply_mask(pred(:,j), box, sp,Wsz);
        f=[coarse_reg2sp double(orig_reg2sp)];
        scr2=refinement_model.spmodel.w(1:end-1)*f'+refinement_model.spmodel.w(end);
        scr2=1./(1+exp(-scr2));
        if(refinement_model.spmodel.Label(1)~=1) scr2=1-scr2; end
        newreg2sp(:,j)=scr2;
    end

    %threshold
    newreg2sp=newreg2sp>=0.5;
    toc;
    %load the gt
    [cls, inst]=load_gt(sbddir, imnames{i});
    [overlap, pprecision, precall]=get_gt_overlaps(newreg2sp, sp, inst);    
    refined_overlaps{i}=overlap;
    refined_pprecision{i}=pprecision;
    refined_precall{i}=precall;
    fprintf('Doing %d\n',i);
end
    
    
    
