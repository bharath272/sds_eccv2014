function boxes=get_region_boxes(sp, reg2sp)
reg2sp=double(reg2sp);
[I,J]=ind2sub(size(sp),[1:numel(sp)]);
numsp=max(sp(:));
X1=accumarray(sp(:), J(:), [numsp 1],@min);
Y1=accumarray(sp(:), I(:), [numsp 1],@min);
X2=accumarray(sp(:), J(:), [numsp 1],@max);
Y2=accumarray(sp(:), I(:), [numsp 1],@max);
Z=reg2sp;
Z(Z==0)=inf;
x1=min(bsxfun(@times, X1, Z),[],1);
y1=min(bsxfun(@times, Y1, Z),[],1);

x1(isinf(x1))=1;
y1(isinf(y1))=1;

x2=max(bsxfun(@times, X2, reg2sp),[],1);
y2=max(bsxfun(@times, Y2, reg2sp),[],1);
x2(x2==0)=1;
y2(y2==0)=1;
boxes=[x1(:) y1(:) x2(:) y2(:)];

	



