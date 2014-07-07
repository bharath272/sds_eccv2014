function [chosen, scores, overlaps]=get_top_regions(chosen_orig, scores_orig, overlaps_all, topn)
%for each category
for i=1:numel(scores_orig)
    fprintf('Pruning category %d\n',i);
    %get all scores
    allscr=cat(1, scores_orig{i}{:});

    %get threshold
    [s1, i1]=sort(allscr, 'descend');
    s1=s1(1:min(numel(i1),topn));    
    thresh=s1(end);


    chosen{i}=cell(numel(chosen_orig{i}),1);
    scores{i}=cell(numel(scores_orig{i}),1);
    overlaps{i}=cell(numel(scores_orig{i}),1);
    for j=1:numel(scores_orig{i})
	    if(isempty(chosen_orig{i}{j})) continue; end
	    chosen{i}{j}=chosen_orig{i}{j}(scores_orig{i}{j}>=thresh);
	    scores{i}{j}=scores_orig{i}{j}(scores_orig{i}{j}>=thresh);
	    overlaps{i}{j}=overlaps_all{j}(:,chosen{i}{j});
    end


end









