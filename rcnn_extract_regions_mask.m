function [batches, masked_batches, batch_padding] = rcnn_extract_regions(im, sp, reg2sp, boxes, rcnn_model)

% convert image to BGR and single
im = single(im(:,:,[3 2 1]));
num_boxes = size(boxes, 1);
batch_size = rcnn_model.cnn.batch_size;
num_batches = ceil(num_boxes / batch_size);
batch_padding = batch_size - mod(num_boxes, batch_size);
if(mod(num_boxes, batch_size)==0) batch_padding=0; end
crop_mode = rcnn_model.detectors.crop_mode;
image_mean = rcnn_model.cnn.image_mean;
crop_size = size(image_mean,1);
crop_padding = rcnn_model.detectors.crop_padding;

batches = cell(num_batches, 1);
masked_batches = cell(num_batches, 1);
for batch = 1:num_batches
%  disp(batch);
%parfor batch = 1:num_batches
  batch_start = (batch-1)*batch_size+1;
  batch_end = min(num_boxes, batch_start+batch_size-1);

  ims = zeros(crop_size, crop_size, 3, batch_size, 'single');
  masked_ims=zeros(crop_size, crop_size, 3, batch_size, 'single');
  for j = batch_start:batch_end
    bbox = boxes(j,:);
    m1=reg2sp(:,j);
    mask=double(m1(sp)); 
    [crop, mask_crop] = rcnn_im_crop_mask(im, mask, bbox, crop_mode, crop_size, ...
        crop_padding, image_mean);
    % swap dims 1 and 2 to make width the fastest dimension (for caffe)
    ims(:,:,:,j-batch_start+1) = permute(crop, [2 1 3]);
    masked_ims(:,:,:,j-batch_start+1) = permute(mask_crop, [2 1 3]);
  end

  batches{batch} = ims;
  masked_batches{batch} = masked_ims;
end
