function visualize_sds(imnames, imgdir, sptextdir, regspimgdir, topchosen, topscores, categid, refineddir)

%only keep this category
topchosen=topchosen{categid};
topscores=topscores{categid};


%sort detections in order of decreasing scores
imids=[];
scrs=[];
detids=[];
for i=1:numel(topchosen)
    imids=[imids; i*ones(numel(topchosen{i}),1)];
    detids=[detids; [1:numel(topchosen{i})]'];
    scrs=[scrs; topscores{i}(:)];
end
[s1, i1]=sort(scrs, 'descend');


%go down the detections and display them
for i=1:numel(i1)
    k=i1(i);
    imid=imids(k);
    detid=detids(k);
    scr=scrs(k);

    %read sprep and maybe refined regions
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{imid} '.txt']), fullfile(regspimgdir, [imnames{imid} '.png']));
    reg2sp=reg2sp(:,topchosen{imid}(detid));
    if(exist('refineddir', 'var'))
        tmp=load(fullfile(refineddir, int2str(categid), [imnames{imid} '.mat']));
        reg2sp=tmp.newreg2sp(:,detid);
    end
    
    %the mask
    m1=reg2sp(sp);

    img=imread(fullfile(imgdir, [imnames{imid} '.jpg']));

    %show
    imagesc(color_seg(double(m1), img)); axis equal;
    title(sprintf('Image %d, detid %d, score %f', imid, detid, scr));
    pause;
end

