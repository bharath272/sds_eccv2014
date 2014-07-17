function models=train_svms_box_onlydetfeats(imnames, feat_dir, region_meta_info, cachedir, run_name, varargin)
%random seed
rng(3);


%input parser to allow additional inputs
p = inputParser;
addOptional(p, 'to_train', [1:20], @isnumeric);
addOptional(p, 'svm_C', 10^-3, @isscalar);
addOptional(p,'bias_mult',10,@isscalar);
addOptional(p, 'pos_loss_weight', 2, @isscalar);
addOptional(p, 'do_latent', 0, @isscalar);
addOptional(p, 'pos_ov_thresh', 0.5, @isscalar);
addOptional(p, 'neg_ov_thresh', 0.3, @isscalar);
addOptional(p, 'hard_neg_thresh', -1.0001, @isscalar);
addOptional(p, 'evict_thresh', -1.2, @isscalar);
addOptional(p, 'retrain_limit', 2000, @isscalar);
parse(p, varargin{:});
opts=p.Results;

to_train=opts.to_train;

%if cachedir does not exist, create it
if(~exist(cachedir, 'file')) mkdir(cachedir); end
cachedir=fullfile(cachedir, run_name);
if(~exist(cachedir, 'file')) mkdir(cachedir); end


%collect feature stats
fprintf('Collecting feature stats...\n');
stats=collect_feature_stats(imnames, feat_dir, cachedir, region_meta_info);

%initialize caches
for i=1:numel(to_train)
    caches(i) = init_cache;
    models(i).categid=to_train(i);
    models(i).w=[];
    models(i).b=[];
    models(i).meannrm=stats.meannrm;
end

%load GT positives
pos_file=fullfile(cachedir, 'pos.mat');
if(exist(pos_file, 'file'))
    tmp=load(pos_file);
    X_pos=tmp.X_pos;
else
    X_pos=get_positives(imnames, feat_dir, to_train, region_meta_info, false, models, opts.pos_ov_thresh);
    save(pos_file, 'X_pos');
end
for i=1:numel(to_train)
    caches(i).X_pos=X_pos{i};
    fprintf('Category %d has %d positives\n', to_train(i), size(X_pos{i},1));
end 

%train
first_time=true;
force_update=false;
max_outer_iter=opts.do_latent+1;
for iter=1:max_outer_iter
    if(iter~=max_outer_iter)
        %subsample negatives
        negidx=randperm(numel(imnames));
        negidx=negidx(1:min(1000, numel(negidx)));
    else
        negidx=1:numel(imnames);
    end
    if(iter>1)
        %latent update
        X_pos=get_positives(imnames, feat_dir, to_train, region_meta_info, true, models, opts.pos_ov_thresh);
        for i=1:numel(to_train)
            caches(i).X_pos=X_pos{i};
            fprintf('Latent update: Category %d has %d positives\n', to_train(i), size(X_pos{i},1));

        end
        force_update=true;
    end
    %for every negative
    for i=1:numel(negidx)
        fprintf('Hard negatives iter: %d, image number: %d/%d\n',iter, i, numel(negidx));
        [X_neg, curr_keys]=get_hard_negatives(negidx(i), imnames, feat_dir, to_train,...
                             region_meta_info, models, {caches.keys}, opts.neg_ov_thresh, opts.hard_neg_thresh, first_time);
       very_last_iter=(iter==max_outer_iter) && (i==numel(negidx));
       for j=1:numel(to_train)
            caches(j).X_neg=cat(1, caches(j).X_neg,X_neg{j});
            caches(j).keys=cat(1, caches(j).keys, curr_keys{j});
            caches(j).num_added=caches(j).num_added+size(X_neg{j},1);
            if(caches(j).num_added>opts.retrain_limit || first_time ||very_last_iter || force_update)
                %time to retrain
                fprintf('Retraining category %d with %d positives and %d negatives\n', to_train(j), size(caches(j).X_pos,1), size(caches(j).X_neg,1));
                models(j)=update_model(models(j), caches(j), opts.svm_C, opts.bias_mult, opts.pos_loss_weight);
                [reg, posloss, negloss, neg_scores]=compute_obj_val(models(j),...
                                                   caches(j), opts.svm_C, opts.pos_loss_weight);
                caches(j).pos_loss(end+1)=posloss;
                caches(j).neg_loss(end+1)=negloss;
                caches(j).reg_loss(end+1)=reg;
                caches(j).tot_loss(end+1)=posloss+negloss+reg;
                for t=1:numel(caches(j).tot_loss)
                    fprintf('%d: %f = %f + %f + %f\n',t,caches(j).tot_loss(t),...
                    caches(j).reg_loss(t),caches(j).pos_loss(t), caches(j).neg_loss(t));
                end

                %evict
                keep=neg_scores>opts.evict_thresh;
                caches(j).X_neg=caches(j).X_neg(keep,:);
                caches(j).keys=caches(j).keys(keep,:);
                fprintf('Kept: %d negatives \n', sum(keep));
                caches(j).num_added=0;
            end
                
        end
        first_time=false;
        force_update=false;
    end        
end

















%--------------------------------
%   Feature stats
%--------------------------------
function stats = collect_feature_stats(imnames, feat_dir, cachedir, region_meta_info)
stat_file=fullfile(cachedir, 'stats.mat');
if(exist(stat_file, 'file'))
    tmp=load(stat_file);
    stats=tmp.stats; clear tmp;
else

    num_images=100;
    regs_per_img=100;
    ri=randperm(numel(imnames));
    ri=ri(1:num_images);
    totalnrm=0;
    totalcnt=0;
    for i=1:num_images
        fprintf('%d/%d\n', i, num_images);

        %load random image
        tmp=load(fullfile(feat_dir, [imnames{ri(i)} '.mat']));

        %pick random regions
        num=min(regs_per_img, region_meta_info.num_regs(ri(i)));
        ri2=randperm(region_meta_info.num_regs(ri(i)));
        ri2=ri2(1:num); 
        
        %get norm
        nrm=sqrt(sum(tmp.feats(ri2,:).^2,2));
        totalnrm=totalnrm+sum(nrm);
        totalcnt=totalcnt+num;
    end
    stats.meannrm=totalnrm./totalcnt;
    save(stat_file, 'stats');
end
fprintf('\n\nMean norm=%f\n\n',stats.meannrm);

%------------------------------
%   Init cache
%------------------------------
function cache=init_cache
cache.X_pos = single([]);
cache.X_neg = single([]);
cache.keys = [];
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];



%------------------------------
%   Get positives from image
%------------------------------

function X_pos=get_positives(imnames, featdir, to_train, region_meta_info, islatent, models, pos_ov_thresh)
for k=1:numel(to_train)
    X_pos{k}=single([]);
end

for i=1:numel(imnames)
    fprintf('Positives: %d/%d\n',i, numel(imnames));
    tmp=load(fullfile(featdir, [imnames{i} '.mat']));
    feats=xform_feat(tmp.feats, models(1).meannrm);
    
    %for every category
    for j=1:numel(to_train)
        sel=[];
        if(islatent)
            %latent update: consider all features greater than thresh
            %find gt for this category and get corresponding overlaps
            gtidx=find(region_meta_info.gt{i}==to_train(j));
            nongt=find(region_meta_info.gt{i}==0);
            
            ovs=region_meta_info.box_overlaps{i}(gtidx,:);
            overlapping=find(max(ovs,[],1)>=pos_ov_thresh);
            ovs=ovs(:,overlapping);
            
            scores=feats*models(j).w+models(j).b;

            %for every gt, find the highest scoring gt that overlaps by >70
            for k=1:numel(gtidx)
                ovs_this=ovs(k,:);
                idx=find(ovs_this>pos_ov_thresh);
                if(isempty(idx)) continue; end
                [m, argmax]=max(scores(nongt(overlapping(idx))));
                sel=[sel nongt(overlapping(idx(argmax)))];
            end
        else
            sel=find(region_meta_info.gt{i}==to_train(j));
        end
        X_pos{j}=cat(1, X_pos{j},feats(sel,:));
    end
end

%------------------------------
%   Hard negatives
%------------------------------

function [X_neg, curr_keys]=get_hard_negatives(i, imnames, featdir, to_train, region_meta_info, models, keys, neg_ov_thresh, score_thresh, first_time)
tmp=load(fullfile(featdir, [imnames{i} '.mat']));
feats=xform_feat(tmp.feats, models(1).meannrm);

if(~first_time)
    %concatenate all models
    w=cat(2,models(:).w);
    b=cat(2,models(:).b);

    %compute all scores
    scores=bsxfun(@plus,feats*w, b);
else
    scores=(score_thresh+1)*ones(size(feats,1), numel(models));
end


%for every category
for j=1:numel(to_train)
    %get the groundtruths that don't belong to this category
    wronggtidx=find(region_meta_info.gt{i}~=to_train(j) & region_meta_info.gt{i}~=0);
    %get the non-groundtruths that overlap by less than threshold
    
    nongt=find(region_meta_info.gt{i}==0);
    gtidx=find(region_meta_info.gt{i}==to_train(j));
    if(~isempty(gtidx))
        idx=find(max(region_meta_info.box_overlaps{i}(gtidx,:),[],1)<neg_ov_thresh);
        nongtidx=nongt(idx);
    else
        nongtidx=nongt;
    end
    allidx=[wronggtidx(:); nongtidx(:)];
    allidx=allidx(scores(allidx,j)>=score_thresh);

    %get the keys for each of these
    tmp_keys=[i*ones(numel(allidx),1) allidx];
    
    %check that duplicate keys are not added
    keep=~ismember(tmp_keys, keys{j}, 'rows');
     
    %add these feats
    X_neg{j}=feats(allidx(keep),:);
    curr_keys{j}=tmp_keys(keep,:);
end


%------------------------------
%   Update model
%------------------------------
function model=update_model(model, cache, svm_c, bias_mult, pos_weight)
opts = sprintf('-w1 %.10f -c %.10f -s 3 -B %.10f', ...
                      pos_weight, svm_c, bias_mult);

X=cat(2, cache.X_pos', cache.X_neg');
y=[ones(size(cache.X_pos,1),1); zeros(size(cache.X_neg,1),1)];
fprintf('Setting second half of feats to 0\n');
d=size(X,1);
X(d/2+1:end,:)=0;

fprintf('calling liblinear with:%s\n',opts);
llm = liblinear_train(y, sparse(double(X)), opts, 'col');
model.w = single(llm.w(1:end-1)');
model.b = single(llm.w(end)*bias_mult);



%---------------------------
% Compute objective value
%---------------------------
function [reg, posloss, negloss, neg_scores]=compute_obj_val(model,cache, svm_c, pos_weight)
pos_scores=cache.X_pos*model.w+model.b;
neg_scores=cache.X_neg*model.w+model.b;
posloss=sum(max(1-pos_scores,0))*svm_c*pos_weight;
negloss=sum(max(1+neg_scores,0))*svm_c;
reg=0.5*model.w'*model.w;


%---------------------------
% xform_feat
%---------------------------
function feats=xform_feat(feats, nrm)
feats=feats*20./nrm;





    
