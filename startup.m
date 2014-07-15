addpath(pwd);
addpath(fullfile(pwd, 'evaluation'));
addpath(fullfile(pwd, 'feature_extractor'));
addpath(fullfile(pwd, 'misc'));
addpath(fullfile(pwd, 'preprocessing'));
addpath(fullfile(pwd, 'refinement'));
addpath(fullfile(pwd, 'region_classification'));
addpath(fullfile(pwd, 'extern/caffe/matlab/caffe'));
addpath(fullfile(pwd, 'evaluation'));
addpath(fullfile(pwd, 'extern/MCG-PreTrained'));
cd extern/MCG-PreTrained;
install;
cd ../liblinear/liblinear-1.94/matlab
make
cd ../../../..
