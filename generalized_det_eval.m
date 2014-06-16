function output=generalized_det_eval(imglist, scoresperimg, overlapsall, gtlabels, categ, low_ov_thresh, high_ov_thresh)
% a complete evaluation that also produces diagnostic information
cnt=0;
for k=1:numel(scoresperimg)
	cnt=cnt+numel(scoresperimg{k});
end

%We will go through and record information about every detection in a big table
%diagnostic=[image_id det_id score label_high dup_high label_low dup_low label_sim dup_sim label_oth dup_oth best_ov best_ov_ind] 
diagnostic=zeros(cnt, 13);
cnt=0;
numgt=0;
 
if(~exist('ovthresh', 'var')) ovthresh=0.5; end
categ_names_and_groups;

for k=1:numel(imglist)
	%print progress
	if(rem(k-1,100)==0) fprintf('Doing : %d/%d\n', k, numel(imglist)); end


	%add things to diagnostic
	numdets=numel(scoresperimg{k});
	diagnostic(cnt+1:cnt+numdets,1)=k;
	diagnostic(cnt+1:cnt+numdets,2)=[1:numdets]';
	diagnostic(cnt+1:cnt+numdets,3)=scoresperimg{k}(:);

		%get all overlaps
	overlaps=overlapsall{k};
	categories=gtlabels{k}(gtlabels{k}~=0);
	
    %record number of ground truth
	numgt=numgt+sum(categories==categ);


    if(isempty(overlaps)) continue; end

	%compute labels using overlaps
	[labels_high, duplicate_high, best_ov, best_ov_ind]=overlaps_to_labels(scoresperimg{k},overlaps, high_ov_thresh, categories, categ);
	[labels_low, duplicate_low]=overlaps_to_labels(scoresperimg{k},overlaps, low_ov_thresh, categories, categ);
	[labels_sim, duplicate_sim]=overlaps_to_labels(scoresperimg{k},overlaps, low_ov_thresh, categories, similar{categ});
	[labels_oth, duplicate_oth]=overlaps_to_labels(scoresperimg{k},overlaps, low_ov_thresh, categories, other{categ});
	assert(sum(labels_low)<=sum(categories==categ));	
	diagnostic(cnt+1:cnt+numdets, 4:end)=[labels_high, duplicate_high, labels_low, duplicate_low, ...
											labels_sim, duplicate_sim, labels_oth, duplicate_oth, ...
											best_ov, best_ov_ind];

	cnt=cnt+numdets;
end

%now that we have accumulated everything, assign blame
%first mislocalization : labels_low | duplicate_low & ~labels_high
mislocerr=(diagnostic(:,6)==1 | diagnostic(:,7)==1) & (diagnostic(:,4)~=1);

%next sim
simerr=(diagnostic(:,8)==1 | diagnostic(:,9)==1) & (~mislocerr) & (diagnostic(:,4)~=1);

%next oth
otherr=(diagnostic(:,10)==1 | diagnostic(:,11)==1) & (~mislocerr) & (~simerr) & (diagnostic(:,4)~=1);
			
%next bg
bgerr=(diagnostic(:,4)~=1) & (~mislocerr) & (~simerr) & (~otherr);

%prec rec
scores=diagnostic(:,3);
labels=diagnostic(:,4);
[ap, prec, rec]=get_ap(scores, labels, numgt);

%prec rec when misloc are removed
scores2=scores;
scores2(mislocerr)=-inf;
[ap_nomisloc, prec_nomisloc, rec_nomisloc]=get_ap(scores2, labels, numgt);
 
%prec rec when misloc are corrected, but duplicates are removed
scores2=scores;
scores2(diagnostic(:,7)==1)=-inf;
[ap_corrmisloc, prec_corrmisloc, rec_corrmisloc]=get_ap(scores2, diagnostic(:,6), numgt);

%prec rec when sim are removed
scores2=scores;
scores2(simerr)=-inf;
[ap_nosim, prec_nosim, rec_nosim]=get_ap(scores2, labels, numgt);

%prec rec when oth are removed
scores2=scores;
scores2(otherr)=-inf;
[ap_nooth, prec_nooth, rec_nooth]=get_ap(scores2, labels, numgt);

%prec rec when bg are removed
scores2=scores;
scores2(bgerr)=-inf;
[ap_nobg, prec_nobg, rec_nobg]=get_ap(scores2, labels, numgt);


%assemble into output struct;
output.diagnostic=diagnostic;
output.mislocerr=mislocerr;
output.simerr=simerr;
output.bgerr=bgerr;
output.otherr=otherr;
output.PR=struct('ap', ap, 'prec', prec, 'rec', rec);
output.PR_nomisloc=struct('ap', ap_nomisloc, 'prec', prec_nomisloc, 'rec', rec_nomisloc);
output.PR_corrmisloc=struct('ap', ap_corrmisloc, 'prec', prec_corrmisloc, 'rec', rec_corrmisloc);
output.PR_nosim=struct('ap', ap_nosim, 'prec', prec_nosim, 'rec', rec_nosim);
output.PR_nooth=struct('ap', ap_nooth, 'prec', prec_nooth, 'rec', rec_nooth);
output.PR_nobg=struct('ap', ap_nobg, 'prec', prec_nobg, 'rec', rec_nobg);


function [labels, duplicate, best_ov, best_ov_ind]=overlaps_to_labels(scores, overlaps,thresh, categories, categ);
%get everything relevant
idx=find(ismember(categories, categ));
overlaps=overlaps(idx,:);

%initialize
covered=false(numel(idx),1);
labels=zeros(numel(scores),1);
duplicate=labels;
best_ov=labels;
best_ov_ind=labels;
if(isempty(idx)) return; end


%sort scores
[s1, i1]=sort(scores, 'descend');


%assign
for k=1:numel(i1)
	if(all(covered)) break; end
	idx22=find(~covered);
	[assign_ov, assign_ind]=max(overlaps(~covered,i1(k)));
	if(assign_ov>thresh) labels(i1(k))=1; 	covered(idx22(assign_ind))=true; end
end

%get the best overlapping, which might be different from assigned
[best_ov, best_ov_ind]=max(overlaps,[],1);
duplicate(best_ov(:)>thresh & labels==0	)=1;
best_ov_ind=idx(best_ov_ind);
best_ov=best_ov(:);
best_ov_ind=best_ov_ind(:);







