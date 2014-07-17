function imagelist_to_seg(imnames, imgdir, mcgdir, ovoutdir, sptextdir, regspimgdir, featdir, scorefile,topk, VOCopts)
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
svmfile='sds_pretrained_models/svms/C.mat';
tmp=load(svmfile);
scores=test_svms(tmp.models, imnames,featdir);

%%%%%%%%%%%%%%%%%%
%NMS and pick top
%%%%%%%%%%%%%%%%%%
fprintf('Picking top detections\n');
[chosen, chosenscores]=region_nms(ovoutdir, imnames, scores, region_meta_info, 20);
%keep only topk per category
[topchosen, topscores]=get_top_regions(chosen, chosenscores, region_meta_info.overlaps, topk);
save(scorefile, 'scores', 'topchosen', 'topscores');

%%%%%%%%%%%%%%%%%%%%
%Paste segmentations
%%%%%%%%%%%%%%%%%%%%
fprintf('Pasting segmentations\n');


tmp=load('sds_pretrained_models/refinement_models.mat');
[local_ids, labels, scores2] = paste_segments(topchosen, scores, region_meta_info, 2, 10, -1);
create_pasted_segmentations_refined('comp6',imnames, local_ids, labels, scores2,VOCopts, sptextdir, regspimgdir, region_meta_info, tmp.refinement_models, featdir, [10 10]);




    
