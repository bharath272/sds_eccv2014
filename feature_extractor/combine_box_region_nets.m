function combine_box_region_nets(box_net, region_net, pi_net)
if(~exist(box_net, 'file'))
    fprintf('Box net does not exist!\n'); return;
end
if(~exist(region_net, 'file'))
    fprintf('Region net does not exist!\n');return;
end
caffe('init', 'prototxts/rcnn_extract_fc7.prototxt', box_net);
box_layers=caffe('get_weights');
caffe('init', 'prototxts/rcnn_extract_fc7.prototxt', region_net);
region_layers=caffe('get_weights');


for i=1:numel(box_layers)
    box_layers(i).layer_names=[box_layers(i).layer_names '_1'];
end
for i=1:numel(box_layers)
    region_layers(i).layer_names=[region_layers(i).layer_names '_2'];
end
layers=[box_layers; region_layers];
caffe('read_prototxt', 'prototxts/pinetwork_extract_fc7.prototxt');
caffe('change_weights', layers);
caffe('save_net', pi_net);

