function demo_sds
imname='2010_002211';
imgdir=pwd;
cachedir=fullfile(pwd, 'cachedir');


imgfile=fullfile(imgdir, [imname '.jpg']);
img=imread(imgfile);


if(~exist(cachedir, 'file')) mkdir(cachedir); end
mcgdir=fullfile(cachedir, 'mcg');
ovoutdir=fullfile(cachedir, 'overlaps');
sptextdir=fullfile(cachedir, 'sptextdir');
regspimgdir=fullfile(cachedir, 'reg2spimgdir');
featdir=fullfile(cachedir, 'featdir');
refinedoutdir=fullfile(cachedir, 'refinement_out');
scorefile=fullfile(cachedir, 'scores.mat');

imagelist_to_sds({imname}, imgdir, mcgdir, ovoutdir, sptextdir, regspimgdir, featdir, refinedoutdir, scorefile,10);
tmp=load(scorefile);
chosen=tmp.topchosen;
chosenscores=tmp.topscores;

%visualize the top detections
visualize_sds({imname}, imgdir, sptextdir, regspimgdir, chosen, chosenscores, 15, refinedoutdir);


