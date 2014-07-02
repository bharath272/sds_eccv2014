function save_sp_without_gt(imnames, mcgdir, sptextdir, regspimgdir)
if(~exist(sptextdir, 'file'))

    mkdir(sptextdir); 
end
if(~exist(regspimgdir, 'file'))
    mkdir(regspimgdir); 
end

N=2000;

for i=1:numel(imnames)
    if(exist(fullfile(sptextdir, [imnames{i} '.txt']), 'file')) continue; end

    tmp=load(fullfile(mcgdir, [imnames{i} '.mat']));
    candidates=tmp.candidates; clear tmp;
    %get reg2sp
    sp=candidates.superpixels;
    N1=min(N, numel(candidates.labels));
    reg2sp=false(max(sp(:)),N1);
    for j=1:N1
        reg2sp(candidates.labels{j},j)=true;
    end
    
    %compress to remove irrelevant sps
    [sp, reg2sp]=compressSp2reg(sp, reg2sp);
    %write sprep
    textfile=fullfile(sptextdir, [imnames{i} '.txt']);
    write_sprep_text(sp, textfile);
    reg2spfile=fullfile(regspimgdir, [imnames{i} '.png']);
    imwrite(uint8(reg2sp),reg2spfile);
    fprintf('Done %d\n', i);
end
        


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
 




function write_sprep_text(sp, filename)
fid=fopen(filename, 'w');
fprintf(fid,'%d %d\n', size(sp,1), size(sp,2));
fprintf(fid, '%d ', sp(:));
fclose(fid);




