function create_pasted_segmentations(name,imnames, local_ids, labels, scores,VOCopts, spdir, regspimgdir, region_meta_info, refinement_models, featdir, Wsz)
for i = 1:numel(imnames)
	[sp, reg2sp]=read_sprep(fullfile(spdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
    d=load(fullfile(featdir, [imnames{i} '.mat']));
    
	cmap=zeros(size(sp));
	[s1, i1]=sort(scores{i}, 'ascend');

    %get nongt
    nongt=find(region_meta_info.gt{i}==0);
    f=d.feats(nongt,:);
    newreg2sp_all=reg2sp;
    for j=1:20
        idx=find(labels{i}==j);
        if(isempty(idx)) continue; end
        idx=local_ids{i}(idx);
        [newreg2sp, sp]=apply_refiner(f(idx,:)', sp, reg2sp(:,idx), refinement_models{j}, Wsz);
        newreg2sp_all(:,idx)=newreg2sp;
    end    
	for j=1:numel(i1)
		m1=newreg2sp_all(:,local_ids{i}(i1(j)));
		m1=m1(sp);
		cmap(logical(m1))=labels{i}(i1(j));
	end
	colmap=VOClabelcolormap;
	resfile = sprintf(VOCopts.seg.clsrespath,name,VOCopts.testset,imnames{i});
	imwrite(cmap+1, colmap, resfile);
	if(rem(i-1,10)==0) disp(i); end	
end 

