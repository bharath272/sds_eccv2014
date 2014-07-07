function feat = rcnn_features_mask(im, sp, reg2sp, boxes, rcnn_model)

% make sure that caffe has been initialized for this model
if rcnn_model.cnn.init_key ~= caffe('get_init_key')
  error('You probably need to call rcnn_load_model');
end

% Each batch contains 256 (default) image regions.
% Processing more than this many at once takes too much memory
% for a typical high-end GPU.
fprintf('Extracting patches...');
tic;
[batches, masked_batches, batch_padding] = rcnn_extract_regions_mask(im, sp, reg2sp, boxes, rcnn_model);
t=toc;
fprintf('[done, %f s]\n', t);
batch_size = rcnn_model.cnn.batch_size;
b1=batches{1}(:,:,:,1);
b2=batches{1}(:,:,:,2);
% compute features for each batch of region images
feat_dim = -1;
feat = [];
curr = 1;
fprintf('Computing features');
tic;
for j = 1:length(batches)
   fprintf('.');
  B=[ batches(j),masked_batches(j)];
  % forward propagate batch of region images 
  f = caffe('forward', B);
  f = f{1};
  % first batch, init feat_dim and feat
  if j == 1
    feat_dim = numel(f)/size(batches{j},4);
    feat = zeros(size(boxes, 1), feat_dim, 'single');
  end

  f = reshape(f, [feat_dim batch_size]);
  % last batch, trim f to size
  if j == length(batches)
    if batch_padding > 0
      f = f(:, 1:end-batch_padding);
    end
  end

  feat(curr:curr+size(f,2)-1,:) = [f'];
  curr = curr + batch_size;
end
t=toc;
fprintf('[done, %fs]\n',t);
