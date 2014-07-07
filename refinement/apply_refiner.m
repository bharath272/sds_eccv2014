function [newreg2sp, sp]=apply_refiner(f, sp, reg2sp, refinement_model, Wsz)


%get boxes
boxes=get_region_boxes(sp, reg2sp);
spf=[];
%for each selected region
for j=1:size(reg2sp,2)
    box=boxes(j,:);
    box=expand_box(box, 16/(227-32));
    msk=reg2sp(:,j);
    msk=msk(sp);
    clipped_msk=clip_img(msk, box);
    cl_msk_rsz=imresize(clipped_msk,Wsz);
    spf=[spf cl_msk_rsz(:)];
end

f=[f; spf];
%make all predictions
scr=bsxfun(@plus, refinement_model.W(:,1:end-1)*f, refinement_model.W(:,end));
pred=1./(1+exp(-scr));    

newreg2sp=zeros(size(reg2sp));
%for each selected region
for j=1:size(reg2sp,2)
    box=boxes(j,:);
    box=expand_box(box, 16/(227-32));
    orig_reg2sp=reg2sp(:,j);
    coarse_reg2sp=apply_mask(pred(:,j), box, sp,Wsz);
    f=[coarse_reg2sp double(orig_reg2sp)];
    scr2=refinement_model.spmodel.w(1:end-1)*f'+refinement_model.spmodel.w(end);
    scr2=1./(1+exp(-scr2));
    if(refinement_model.spmodel.Label(1)~=1) scr2=1-scr2; end
    newreg2sp(:,j)=scr2;
end

%threshold
newreg2sp=newreg2sp>=0.5;



