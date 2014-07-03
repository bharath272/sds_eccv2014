function [overlap, pprecision, precall]=get_gt_overlaps(reg2sp, sp, instimg)
if(all(instimg==0)) overlap=zeros(0,size(reg2sp,2)); return; end
spareas=accumarray(sp(:),1);
totalareas=spareas'*reg2sp;
instimg=double(instimg);
insts=unique(instimg(instimg~=0));
intsp=accumarray([sp(instimg~=0) instimg(instimg~=0)], 1, [max(sp(:)) max(insts(:))]);
assert(size(intsp,1)==max(sp(:)));
int=intsp'*reg2sp;
instareas=sum(intsp,1);
uni=bsxfun(@plus, totalareas(:)', sum(intsp,1)')-int;
overlap=int./uni;
pprecision=bsxfun(@rdivide, int, totalareas(:)');
precall=bsxfun(@rdivide, int, sum(intsp,1)');
pprecision(isnan(pprecision))=0;
precall(isnan(precall))=0;
