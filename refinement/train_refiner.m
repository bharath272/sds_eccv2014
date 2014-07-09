function refinement_model=train_refiner(imnames, region_meta_info, featdir, sptextdir, regspimgdir, sbddir, Wsz, categid)
feats=[];
labels=[];

regidsall=cell(numel(imnames),1);
movindall=cell(numel(imnames),1);
%collect training examples
for i=1:numel(imnames)
    %check if there is any groundtruth for this category
    if(~any(region_meta_info.gt{i}==categid))
        continue;
    end
    fprintf('Getting training regions %d\n', i);

    %find ground truths
    gtids=find(region_meta_info.gt{i}==categid);

    %find regions that overlap by more than 70\%
    [mov, movind]=max(region_meta_info.overlaps{i}(gtids,:),[],1);
    regids=find(mov>=0.7);
    movind=gtids(movind(regids));
    
    %index into features/sp
    nongtids=find(region_meta_info.gt{i}==0);
    regids=nongtids(regids);

    regidsall{i}=regids;
    movindall{i}=movind;
end
for i=1:numel(imnames)
    regids=regidsall{i};
    movind=movindall{i};
    if(isempty(regids))
        continue;
    end
    fprintf('Getting training features:%d\n', i);

    %load gt
    [cls, inst] = load_gt(sbddir, imnames{i});
    %load features
    d=load(fullfile(featdir, [imnames{i} '.mat']));
    f=d.feats(regids,:)';

    %load superpixels
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
       
    %get boxes
    boxes=get_region_boxes(sp, reg2sp);
    spf=[];
    lab=[];
    %for each selected region
    for j=1:numel(regids)
        box=boxes(regids(j),:);
        box=expand_box(box, 16/(227-32));
        msk=reg2sp(:,regids(j));
        msk=msk(sp);
        clipped_msk=clip_img(msk, box);
        clipped_gt=clip_img(double(double(inst)==movind(j)),box);
        cl_msk_rsz=imresize(clipped_msk,Wsz);
        cl_gt_rsz=imresize(clipped_gt, Wsz);
        spf=[spf cl_msk_rsz(:)];
        lab=[lab cl_gt_rsz(:)>0.5];
    end
    
    f=[f; spf];
    if(isempty(feats))
        feats=zeros(size(f,1),10000);
        labels=zeros(10000,prod(Wsz));
        cnt=0;
    end
    feats(:,cnt+(1:size(f,2)))=f;
    labels(cnt+(1:size(f,2)),:)=double(lab');
    cnt=cnt+size(f,2);
end
feats=feats(:,1:cnt);
labels=labels(1:cnt,:);

%train svms
for i=1:prod(Wsz)
fprintf('Training coarse model for grid cell %d\n',i);
logisticmodel=liblinear_train(double(labels(:,i)), double(feats), '-s 0 -c 0.01 -B 1', 'col');
if(logisticmodel.Label(1)==0)
    W{i}=-logisticmodel.w;
else
    W{i}=logisticmodel.w;
end
end
W=cat(1, W{:});

%make all predictions
scr=bsxfun(@plus, W(:,1:end-1)*feats, W(:,end));
pred=1./(1+exp(-scr));    


%train superpixel predictor
feats=zeros(2,1000000);
labels=zeros(1, 1000000);;
cnt=0;
cnt2=0;
for i=1:numel(imnames)
    regids=regidsall{i};
    movind=movindall{i};
    if(isempty(regids)) continue; end    
    fprintf('Getting second stage feats:%d\n',i);
    %load superpixels
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
   
    %get all coarse predictions
    predcurr=pred(:,cnt+(1:numel(regids)));
    cnt=cnt+numel(regids);

    boxes=get_region_boxes(sp, reg2sp);

    spareas=accumarray(sp(:),1);
    %apply it to each region
    for j=1:numel(regids)
        box=boxes(regids(j),:);
        box=expand_box(box, 16/(227-32));
        coarse_reg2sp=apply_mask(predcurr(:,j), box, sp,Wsz);
        orig_reg2sp=reg2sp(:,regids(j));
        
        %only choose large pixels and pixels for which prediction is large enough
        idx=find(spareas>100 & (coarse_reg2sp>0.2 | orig_reg2sp>0));


        %next get the true labels
        lab=reg2sp(:,movind(j));
        feats(1,cnt2+(1:numel(idx)))=coarse_reg2sp(idx);
        feats(2, cnt2+(1:numel(idx)))=orig_reg2sp(idx);
        labels(cnt2+(1:numel(idx)))=lab(idx);
        cnt2=cnt2+numel(idx);
    end
end
feats=feats(:,1:cnt2);
labels=labels(:,1:cnt2);

%learn second model
spmodel=liblinear_train(labels(:), feats, '-s 0 -c 1 -B 1', 'col');

refinement_model.W=W;
refinement_model.spmodel=spmodel;        
