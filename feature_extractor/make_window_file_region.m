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
  fprintf(fid, '%s\n', fullfile(regspimgdir, [imnames{i} '.png']));
  fprintf(fid, '%s\n', fullfile(sptextdir, [imnames{i} '.txt']));
  fprintf(fid, '%d\n%d\n%d\n', ...
      prec.size.depth, ...
      prec.size.height, ...
      prec.size.width);
  fprintf(fid, '%d\n', num_boxes);
  gtidx=find(region_meta_info.gt{i}~=0);
  nongtidx=find(region_meta_info.gt{i}==0);
  for j=1:numel(gtidx)
    ov=1;
    label=region_meta_info.gt{i}(gtidx(j));
    bbox = region_meta_info.boxes{i}(gtidx(j),:)-1;
    fprintf(fid, '%d %.3f %d %d %d %d %d\n', ...
        label, ov, bbox(1), bbox(2), bbox(3), bbox(4), gtidx(j));
  end
  for j=1:numel(nongtidx)
    [ov,ovidx]=max(region_meta_info.overlaps{i}(:,j));
    if(ov>=1e-5)
        label=region_meta_info.gt{i}(gtidx(ovidx));
    else
        ov=0;
        label=0;
    end
    bbox = region_meta_info.boxes{i}(nongtidx(j),:)-1;
    fprintf(fid, '%d %.3f %d %d %d %d %d\n', ...
        label, ov, bbox(1), bbox(2), bbox(3), bbox(4), nongtidx(j));
  end
end

fclose(fid);
