function overlaps=get_overlaps_for_selected(region_meta_info, chosen)
for i=1:numel(chosen)
overlaps{i}=region_meta_info.overlaps{i}(:,chosen{i});
end
