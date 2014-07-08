function demo_sds(imname,imgdir, cachedir)
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



%find the top detections
detids=[];
categids=[];
scr=[];
for i=1:20
    detids=[detids;[1:numel(chosen{i}{1})]'];
    scr=[scr; chosenscores{i}{1}(:)];
    categids=[categids; i*ones(numel(chosen{i}{1}),1)];
end
[s1,i1]=sort(scr, 'descend');


%visualize the top detections
categ_names_and_groups;
[sp, reg2sp]=read_sprep(fullfile(sptextdir, [imname '.txt']), fullfile(regspimgdir, [imname '.png']));
for k=1:numel(i1)
    categid=categids(i1(k));
    detid=detids(i1(k));
    score=scr(i1(k));
    M1=reg2sp(:,chosen{categid}{1}(detid)); M1=M1(sp);
    subplot(1,2,1); imagesc(color_seg(double(M1),img)); axis equal; title(sprintf('score:%f, category:%s',score, categnames{categid}));
    tmp=load(fullfile(cachedir, 'refinement_out',int2str(categid),[imname '.mat']));
    M1=tmp.newreg2sp(:,detid); M1=M1(sp);
    subplot(1,2,2); imagesc(color_seg(double(M1),img)); axis equal;
   disp(k);
    pause; 
end




