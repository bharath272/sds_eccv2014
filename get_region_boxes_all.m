function boxes=get_region_boxes_all(imnames, sptextdir, regspimgdir)
for i=1:numel(imnames)
    [sp, reg2sp]=read_sprep(fullfile(sptextdir, [imnames{i} '.txt']), fullfile(regspimgdir, [imnames{i} '.png']));
    boxes{i}=get_region_boxes(sp, reg2sp);
    fprintf('Doing %d\n', i);
end
