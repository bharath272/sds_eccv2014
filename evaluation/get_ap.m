function [ap, prec, rec, s1]=get_ap(scores, labels, numgt,areas)
scores=scores(:);
labels=labels(:);
if(~exist('areas', 'var'))
	areas=ones(size(labels));
end

numgtd=exist('numgt', 'var');
if(numgtd)
	numgtd=numgt~=0;
end


[s1, i1]=sort(scores, 'descend');
tp=labels(i1);
fp=1-labels(i1);
tp=tp.*areas(i1);
fp=fp.*areas(i1);
tp=cumsum(tp);
fp=cumsum(fp);
prec=tp./(tp+fp);
if(numgtd)
rec=tp./numgt;
else
rec=tp./sum(labels.*areas);
end
ap=VOCap(rec,prec);
