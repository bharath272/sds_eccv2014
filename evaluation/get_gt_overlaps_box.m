function box_overlaps=get_gt_overlap_box(reg2sp, sp, instimg)
if(all(instimg==0)) box_overlaps=zeros(0,size(reg2sp,2)); return; end
boxes=get_region_boxes(sp, reg2sp);
instimg=double(instimg);
insts=unique(instimg(instimg~=0));
gt_boxes=zeros(numel(insts),4);
for i=1:numel(insts)
    [I,J]=find(instimg==insts(i));
    I=I(:); J=J(:);
    gt_boxes(i,:)=[min(J) min(I) max(J) max(I)];
end
boxes(:,3:4)=boxes(:,3:4)-boxes(:,1:2)+1;
gt_boxes(:,3:4)=gt_boxes(:,3:4)-gt_boxes(:,1:2)+1;
int=rectint(boxes, gt_boxes);
uni=bsxfun(@plus, prod(boxes(:,3:4),2), prod(gt_boxes(:,3:4),2)')-int;
box_overlaps=int./(uni+double(uni~=0));
box_overlaps=box_overlaps';

