function [region_meta_info]=preprocess_cpmc_candidates(imnames, cpmcdir, gtdir, ovoutdir, N)
if(~exist(ovoutdir, 'file'))
    mkdir(ovoutdir); 
end


for i=1:numel(imnames)
    %load sprep
    x1=load(fullfile(cpmcdir, [imnames{i} '.mat']));
    sp=x1.sp;
    reg2sp=x1.sp_app;
    
    %compute overlaps between everything
    ov=double(reg2sp')*double(reg2sp);
    ov=ov>0;
    if(~exist(fullfile(ovoutdir, [imnames{i} '.mat']), 'file'))
    save(fullfile(ovoutdir, [imnames{i} '.mat']), 'ov');
    end
    %load gt
    if(isempty(gtdir))
        %no gt
        inst=zeros(size(sp));
        categories=[];
    else
    [cls, inst, categories]=load_gt(gtdir, imnames{i});
    end
    %compute overlaps with ground truth
    overlap=get_gt_overlaps(reg2sp, sp, inst);    
    box_overlap=get_gt_overlaps_box(reg2sp, sp, inst);
    %populate meta info
    region_meta_info.num_regs(i)=size(reg2sp,2);
    region_meta_info.overlaps{i}=overlap;
    region_meta_info.box_overlaps{i}=box_overlap;
    region_meta_info.gt{i}=[zeros(1,size(reg2sp,2))];
    fprintf('Done %d\n', i);
end
        







function write_sprep_text(sp, filename)
fid=fopen(filename, 'w');
fprintf(fid,'%d %d\n', size(sp,1), size(sp,2));
fprintf(fid, '%d ', sp(:));
fclose(fid);









function [sp, sp2reg]=intersect_partitions(sp1, sp2, sp2reg1, sp2reg2)
%do a unique
[unique_pairs, junk, sp]=unique([sp1(:) sp2(:)], 'rows');
sp=reshape(sp, size(sp1));

%create new sp2reg
sp2reg=[sp2reg1(unique_pairs(:,1),:) sp2reg2(unique_pairs(:,2),:)];


%break disconnected sp
 spNew = zeros(size(sp));
  sp2regNew = false(size(sp2reg));
  cnt = 0;
  %% Check connectivity of superpixels
  for i = 1:max(sp(:))
    tt = bwconncomp(sp == i);
    for j = 1:tt.NumObjects,
      cnt = cnt+1;
      spNew(tt.PixelIdxList{j}) = cnt;
      sp2regNew(cnt, :) = sp2reg(i, :);
    end
  end

sp=spNew;
sp2reg=sp2regNew;

function [spNew sp2regNew] = compressSp2reg(sp, sp2reg)
  assert(islogical(sp2reg), 'sp2reg should be logical');
  [sp2reg bb spSame] = unique(sp2reg, 'rows');
  sp = spSame(sp);

  spNew = zeros(size(sp));
  sp2regNew = false(size(sp2reg));
  cnt = 0;
  %% Check connectivity of superpixels
  for i = 1:max(sp(:))
    tt = bwconncomp(sp == i);
    for j = 1:tt.NumObjects,
      cnt = cnt+1;
      spNew(tt.PixelIdxList{j}) = cnt;
      sp2regNew(cnt, :) = sp2reg(i, :);
    end
  end
 
