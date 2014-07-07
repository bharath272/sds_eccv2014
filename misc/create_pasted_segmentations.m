function create_pasted_segmentations(name,imnames, local_ids, labels, scores,VOCopts, spdir, regspimgdir, region_meta_info)
for i = 1:numel(imnames)
	[sp, reg2sp]=read_sprep(fullfile(spdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
	cmap=zeros(size(sp));
	[s1, i1]=sort(scores{i}, 'ascend');

    %get nongt
    nongt=find(region_meta_info.gt{i}==0);
    
	for j=1:numel(i1)
		m1=reg2sp(:,nongt(local_ids{i}(i1(j))));
		m1=m1(sp);
		cmap(logical(m1))=labels{i}(i1(j));
	end
	colmap=VOClabelcolormap;
	resfile = sprintf(VOCopts.seg.clsrespath,name,VOCopts.testset,imnames{i});
	imwrite(cmap+1, colmap, resfile);
	if(rem(i-1,100)==0) disp(i); end	
end 

