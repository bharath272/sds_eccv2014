function [segm_ids, labels, their_scores] = nms_inference_simplicity_bias(chosen, scores, MAX_OVER, MAX_SEGMS, SIMP_FACTOR, background_thresh,return_bground)
  % the basic idea is that it should be easier adding a first
  % non-background segment than two, and easier two than three, etc.

  DefaultVal('*SIMP_FACTOR', '0.02'); % threshold increases by SIMP_FACTOR from k to k+1 rank

  DefaultVal('*MAX_OVER', '0.5');
  DefaultVal('*MAX_SEGMS', 'inf');
  DefaultVal('*return_bground', 'false');
  
  BACKGROUND = 21; % last one assumed to be background
  
    
  %n = cellfun(@numel, nPBM.img_2_whole_ids(img_ids));  
  
  scores_cell=scores;
  segm_ids = cell(numel(scores_cell),1);
  their_scores = segm_ids;
  labels = segm_ids;

  
  %parfor i=1:numel(scores_cell)  
  for i=1:numel(scores_cell)
	if(rem(i-1,10)==0) fprintf('.'); end
	scores_cell{i}(:,end+1)=background_thresh;
    bground_score = background_thresh;%max(scores_cell{i}(BACKGROUND,:));
    [img_scores, these_labels] = max(scores_cell{i},[],2);
	img_scores=img_scores';
	these_labels=these_labels';


	the_I=chosen{i};
    [s1, i1]=sort(img_scores(the_I), 'descend');
	the_I=the_I(i1);    
    if(MAX_SEGMS~=inf)
              % MAX number of any segments
        the_I = the_I(1:min(numel(the_I),MAX_SEGMS));
    end
    
    ;
    
    if(~return_bground)
      % retain only non-background segments    
      the_I(these_labels(the_I)==BACKGROUND) = [];
    end
    
    % simplicity bias
    to_keep = (img_scores(the_I) > [bground_score + [0 SIMP_FACTOR*(1:((numel(the_I)-1)))]]);
    the_I = the_I(to_keep);
    
    segm_ids{i} = the_I;
    
    their_scores{i} = img_scores(the_I);
    labels{i} = these_labels(the_I);
    
    end
	fprintf('\n');
end
