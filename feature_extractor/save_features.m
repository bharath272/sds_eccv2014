function save_features(imnames, imgdir, sptextdir, regspimgdir, outdir, rcnn_model)
if(~exist(outdir, 'file'))
    fprintf('Creating directory\n');
    mkdir(outdir);
end
for i=1:numel(imnames)
    fprintf('Doing %d/%d\n',i, numel(imnames));
    if(exist(fullfile(outdir, [imnames{i} '.mat']),'file'))
        continue;
    end
    
    %read image and superpixels
    img=imread(fullfile(imgdir, [imnames{i} '.jpg']));
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
    boxes=get_region_boxes(sp, reg2sp);
    
    %pass it through rcnn
    feats=rcnn_features_pi(img, sp, reg2sp, boxes, rcnn_model);

    %save features
    save(fullfile(outdir, [imnames{i} '.mat']), 'feats');
end
