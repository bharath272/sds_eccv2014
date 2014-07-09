function region_meta_info = setup_svm_training(imnames, imgdir, mcgdir, ovoutdir, sptextdir, regspimgdir, featdir, sbddir)
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
[region_meta_info]=preprocess_mcg_candidates(imnames, mcgdir, sbddir, ovoutdir, sptextdir, regspimgdir, 2000);

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


