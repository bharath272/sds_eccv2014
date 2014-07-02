function write_test_boxes(imnames, outpath, id,region_meta_info, topchosen, topscores)
categnames={'aeroplane';'bicycle';'bird';'boat';'bottle';'bus';'car';'cat';'chair';'cow';'diningtable';'dog';'horse';'motorbike';'person';'pottedplant';'sheep';'sofa';'train';'tvmonitor'};
for j=1:20
	filename=sprintf(outpath, id, categnames{j});
    fid=fopen(filename, 'w');
	cnt=0;
	for i=1:numel(imnames)
        nongt=find(region_meta_info.gt{i}==0);
        boxes=region_meta_info.boxes{i}(nongt(topchosen{j}{i}),:);
	    boxes(:,end+1)=topscores{j}{i}(:);
		cnt=cnt+size(boxes,1);
		for k=1:size(boxes,1)
			fprintf(fid, '%s %f %d %d %d %d\n', imnames{i}, boxes(k,end), boxes(k,1:4));
		end
	end
	fclose(fid);
	disp(j);
	disp(cnt);
end
