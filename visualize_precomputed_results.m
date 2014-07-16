function visualize_precomputed_results(precomputed_results_dir, imgdir, categid)
%read the image list
fid=fopen('val.txt');
names=textscan(fid, '%s');
names=names{1};

sptextdir=fullfile(precomputed_results_dir, 'sprep', 'sptextdir');
regspimgdir=fullfile(precomputed_results_dir, 'sprep', 'regspimgdir');
refineddir=fullfile(precomputed_results_dir, 'refined_outdir');
tmp=load(fullfile(precomputed_results_dir, 'postnmsscores.mat'));
visualize_sds(names, imgdir, sptextdir, regspimgdir, tmp.topchosen, tmp.topscores, categid, refineddir);


