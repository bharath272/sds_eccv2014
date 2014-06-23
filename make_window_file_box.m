function make_window_file_region(imnames, region_meta_info, VOCopts, sptextdir, regspimgdir, window_file)
% window_file format
%  # image_index 
%  img_path
%  channels 
%  height 
%  width
%  num_windows
%  class_index overlap x1 y1 x2 y2

fid = fopen(window_file, 'wt');

for i = 1:numel(imnames)
  fprintf('rec %d/%d\n', i, numel(imnames));
  img_path = sprintf(VOCopts.imgpath, imnames{i});
  prec = PASreadrecord(sprintf(VOCopts.annopath, imnames{i}));

  num_boxes = region_meta_info.num_regs(i);
  fprintf(fid, '# %d\n', i-1);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', ...
      prec.size.depth, ...
      prec.size.height, ...
      prec.size.width);
  gtidx=find(region_meta_info.gt{i}~=0);
  nongtidx=find(region_meta_info.gt{i}==0);
  %dedup
  idx=dedup_boxes(region_meta_info.boxes{i}(nongtidx,:));
  fprintf(fid, '%d\n', numel(gtidx)+numel(idx));



  for j=1:numel(gtidx)
    ov=1;
    label=region_meta_info.gt{i}(gtidx(j));
    bbox = region_meta_info.boxes{i}(gtidx(j),:)-1;
    fprintf(fid, '%d %.3f %d %d %d %d\n', ...
        label, ov, bbox(1), bbox(2), bbox(3), bbox(4));
  end
  for k=1:numel(idx)
    j=idx(k);
    %only save boxes that survived deduplication
    [ov,ovidx]=max(region_meta_info.box_overlaps{i}(:,j));
    if(ov>=1e-5)
        label=region_meta_info.gt{i}(gtidx(ovidx));
    else
        ov=0;
        label=0;
    end
    bbox = region_meta_info.boxes{i}(nongtidx(j),:)-1;
    fprintf(fid, '%d %.3f %d %d %d %d\n', ...
        label, ov, bbox(1), bbox(2), bbox(3), bbox(4));
  end
end

fclose(fid);


function idx=dedup_boxes(boxes)
bx=boxes;
bx(:,3:4)=bx(:,3:4)-bx(:,1:2)+1;
area=prod(bx(:,3:4),2);

int=rectint(bx, bx);
uni=bsxfun(@plus, area, area')-int;
ov=int./(uni+double(uni==0));
dist=1-ov;


Y=squareform(dist);
Z=linkage(Y, 'complete');
T=cluster(Z, 'cutoff', 0.05, 'criterion', 'distance');
[junk, ia]=unique(T);
idx=ia;
