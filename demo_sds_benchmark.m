function demo_sds_benchmark(imgdir, cachedir, sbddir)

fid=fopen('val_debug.txt');
imnames=textscan(fid, '%s');
imnames=imnames{1};
if(~exist(cachedir, 'file')) mkdir(cachedir); end
mcgdir=fullfile(cachedir, 'mcg');
ovoutdir=fullfile(cachedir, 'overlaps');
sptextdir=fullfile(cachedir, 'sptextdir');
regspimgdir=fullfile(cachedir, 'reg2spimgdir');
featdir=fullfile(cachedir, 'featdir');
refinedoutdir=fullfile(cachedir, 'refinement_out');
scorefile=fullfile(cachedir, 'scores.mat');

imagelist_to_sds(imnames, imgdir, mcgdir, ovoutdir, sptextdir, regspimgdir, featdir, refinedoutdir, scorefile,2000);
tmp=load(scorefile);
chosen=tmp.topchosen;
chosenscores=tmp.topscores;


output=run_benchmark(imnames, chosen, chosenscores, sptextdir, regspimgdir, sbddir, 15);
pause;
output=run_benchmark(imnames, chosen, chosenscores, sptextdir, regspimgdir, sbddir, 15, refinedoutdir);

