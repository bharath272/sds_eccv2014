function [local_ids, labels, scores2, THRESH]=paste_segments(chosenpercateg,scores, region_meta_info,MAX_AVG_N_SEGM, NMS_MAX_SEGMS, THRESH)
n_segms_per_img = inf;

    
NMS_MAX_OVER = 0; % spatial non-maximum supression (1 means nms is not performed)

SIMP_BIAS_STEP = 0;%0.02; % background threshold increase for each additional segment above 1

% max number of segments per image on average (used to set the background threshold)
% we set this value to the average number of objects in the
% training set. Of course, that is just a coincidence. ;-)
%MAX_AVG_N_SEGM = 2.2; 
for i=1:numel(scores)
    scores{i}=scores{i}(region_meta_info.gt{i}==0,:);
    for j=1:20
        scores{i}(~ismember([1:size(scores{i},1)], chosenpercateg{j}{i}),j)=-inf;
    end
end




chosen=chosenpercateg{1};
for k=2:20
    for i=1:numel(chosen)
        chosen{i}=[chosen{i}(:); chosenpercateg{k}{i}(:)];
    end
end
for i=1:numel(chosen)
    chosen{i}=unique(chosen{i});
end    
while(n_segms_per_img > MAX_AVG_N_SEGM)

    [local_ids, labels, scores2] = nms_inference_simplicity_bias(chosen, scores, NMS_MAX_OVER, NMS_MAX_SEGMS, SIMP_BIAS_STEP, THRESH);                   

    n_segms_per_img = numel(cell2mat(labels')) / numel(labels)

    THRESH = THRESH                
    THRESH = THRESH+0.01;
end
THRESH=THRESH-0.01;


