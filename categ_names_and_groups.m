categnames={'aeroplane';'bicycle';'bird';'boat';'bottle';'bus';'car';'cat';'chair';'cow';'diningtable';'dog';'horse';'motorbike';'person';'pottedplant';'sheep';'sofa';'train';'tvmonitor'};
transport=[1 2 4 6 7 14 19];
artic=[3 8 10 12 13 15 17];
indoor=[5 9 11 16 18 20];

groups={transport, artic, indoor};
for i=1:numel(groups)
for k=1:numel(groups{i})
	similar{groups{i}(k)}=setdiff(groups{i}, groups{i}(k));
end
end
for i=1:20
other{i}=setdiff([1:20], similar{i});
end



