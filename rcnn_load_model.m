function rcnn_model = rcnn_load_model(rcnn_model_or_file, use_gpu)
%  model = 
%    cnn: [1x1 struct]
%        binary_file: 'path/to/cnn/model/binary'
%        definition_file: 'path/to/cnn/model/definition'
%        batch_size: 256
%        image_mean: [227x227x3 single]
%        init_key: -1
%    detectors.W: [N x <numclasses> single]  % matrix of SVM weights
%    detectors.B: [1 x <numclasses> single]  % (row) vector of SVM biases
%    detectors.class_to_index: map from class name to column index in W
%    detectors.crop_mode: 'warp' or 'square'
%    detectors.crop_padding: 16
%    detectors.nms_thresholds: [1x20 single]
%    detectors.training_opts: [1x1 struct]
%        bias_mult: 10
%        TODO(rm): clss: {20x1 cell}
%        fine_tuned: 1
%        layer: 'fc7'
%        pos_loss_weight: 2
%        svm_C: 1.0000e-03
%        trainset: 'trainval'
%        use_flipped: 0
%        year: '2007'
%        feat_norm_mean: 20.1401
%    classes: {cell array of class names}

if isstr(rcnn_model_or_file)
  assert(logical(exist(rcnn_model_or_file, 'file')));
  ld = load(rcnn_model_or_file);
  rcnn_model = ld.rcnn_model; clear ld;
else
  rcnn_model = rcnn_model_or_file;
end

rcnn_model.cnn.init_key = ...
    caffe('init', rcnn_model.cnn.definition_file, rcnn_model.cnn.binary_file);
if exist('use_gpu', 'var') && ~use_gpu
  caffe('set_mode_cpu');
else
  caffe('set_mode_gpu');
end
caffe('set_phase_test');
rcnn_model.cnn.layers = caffe('get_weights');
