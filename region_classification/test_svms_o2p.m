function scores=test_svms(models, imnames,featdir)
W=cat(2, models.w);
b=cat(2, models.b);
for i=1:numel(imnames)
    feats=load_feats(featdir, imnames{i});
    feats=xform_feat(feats, models(1).meannrm);
    scores{i}=bsxfun(@plus, feats*W,b);
    fprintf('Done %d/%d\n',i, numel(imnames));
end

function feats=xform_feat(feats, nrm)
feats=feats*20./nrm;

function feats=load_feats(featdir, image_id)
x1=load(fullfile(featdir,'CPMC_segms_150_sp_approx_LBP_f_pca_2500_noncent',[image_id '.mat']));
x2=load(fullfile(featdir,'CPMC_segms_150_sp_approx_SIFT_GRAY_f_g_pca_5000_noncent', [image_id '.mat']));
x3=load(fullfile(featdir,'CPMC_segms_150_sp_approx_SIFT_GRAY_mask_pca_5000_noncent', [image_id '.mat']));
 feats= [x1.D;x2.D; x3.D]';






