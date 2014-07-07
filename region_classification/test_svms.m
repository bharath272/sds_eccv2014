function scores=test_svms(models, imnames,featdir)
W=cat(2, models.w);
b=cat(2, models.b);
for i=1:numel(imnames)
    tmp=load(fullfile(featdir,[imnames{i} '.mat']));
    feats=xform_feat(tmp.feats, models(1).meannrm);
    scores{i}=bsxfun(@plus, feats*W,b);
    fprintf('Done %d/%d\n',i, numel(imnames));
end

function feats=xform_feat(feats, nrm)
feats=feats*20./nrm;





