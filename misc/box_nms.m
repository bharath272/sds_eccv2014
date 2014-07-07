function [chosen, chosenscores]=box_nms(imnames, scores, region_meta_info, numcats)
chosen=cell(numcats,1);
chosenscores=cell(numcats,1);
for i=1:numel(chosen)
    chosen{i}=cell(numel(imnames),1);
    chosenscores{i}=cell(numel(imnames),1);
end
for k=1:numel(imnames)

    
    %prune out gt
    nongt=find(region_meta_info.gt{k}==0);
    scr=scores{k};
    scr=scr(nongt,:);

    %get all overlaps
    bx=region_meta_info.boxes{k}(nongt,:);
    bx(:,3:4)=bx(:,3:4)-bx(:,1:2)+1;
    int=rectint(bx, bx);
    areas=prod(bx(:,3:4),2);
    uni=bsxfun(@plus, areas, areas')-int;
    ov=int./(uni+double(uni==0));
    ov=double(ov>=0.3);
    

	%for every category
	for i=1:size(scr,2)
	    [s1,i1]=sort(scr(:,i), 'descend');
	    i1=i1(~isinf(s1));	


		chosen1=[];
		while(~isempty(i1))
			pick=i1(1);
						
			chosen1=[chosen1 pick];
			idx=ov(pick, i1)>0;
			idx(1)=true;
			i1(idx)=[];
		end
	    chosen{i}{k}=chosen1;
	    chosenscores{i}{k}=scr(chosen1,i);
	end
	if(rem(k-1,10)==0) fprintf('Doing %d/%d\n', k, numel(imnames)); end
end
