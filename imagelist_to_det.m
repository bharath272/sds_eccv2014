function imagelist_to_det(imnames, imgdir, mcgdir, ovoutdir, sptextdir, regspimgdir, featdir, scorefile,topk, VOCopts)
%%%%%%
%MCG
%%%%%%
fprintf('Computing MCG. Note that this can be parallelized\n');

if(~exist(mcgdir, 'file')) mkdir(mcgdir); end

for i=1:numel(imnames)
    fprintf('MCG : %d/%d\n', i, numel(imnames));
    imname=imnames{i};
    imgfile=fullfile(imgdir, [imname '.jpg']);
    img=imread(imgfile);
    candidates_file=fullfile(mcgdir,[imname '.mat']);
    if(~exist(candidates_file))
    
        candidates=im2mcg(img, 'accurate');
        save(candidates_file, 'candidates');
    end
end

%%%%%%%%%%%
%Preprocess
%%%%%%%%%%%
fprintf('Preprocessing candidates.\n');
[region_meta_info]=preprocess_mcg_candidates(imnames, mcgdir, [], ovoutdir, sptextdir, regspimgdir, 2000);

%%%%%%%%%%%%%
%Load network
%%%%%%%%%%%%%
model_def_file='prototxts/pinetwork_extract_fc7.prototxt';
model_file='sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);

%%%%%%%%%%%%%%%%%
%Extract features
%%%%%%%%%%%%%%%%%
fprintf('Saving features..\n');
save_features(imnames, imgdir, sptextdir, regspimgdir, featdir, rcnn_model);

%%%%%%%%%%%%%%%%%
%Score candidates
%%%%%%%%%%%%%%%%%
fprintf('Scoring using SVMs..\n');
svmfile='sds_pretrained_models/svms_box/C.mat';
tmp=load(svmfile);
scores=test_svms(tmp.models, imnames,featdir);

%%%%%%%%%%%%%%%%%%
%NMS and pick top
%%%%%%%%%%%%%%%%%%
fprintf('Picking top detections\n');
[chosen, chosenscores]=box_nms(imnames, scores, region_meta_info, 20);
%keep only topk per category
[topchosen, topscores]=get_top_regions(chosen, chosenscores, region_meta_info.box_overlaps, topk);
save(scorefile, 'scores', 'topchosen', 'topscores');

%%%%%%%%%%%%%%%%%%%%%%%%
%Write detections
%%%%%%%%%%%%%%%%%%%%%%%
if(~exist(VOCopts.resdir, 'file'))
    mkdir(VOCopts.resdir);
    mkdir(fullfile(VOCopts.resdir, 'Main'));
end
write_test_boxes(imnames, VOCopts.detrespath, 'comp4',region_meta_info, topchosen, topscores);









    
