function box=expand_box(box, frac)
wh=box(3:4)-box(1:2)+1;
padding=wh*frac;
box=box+[-padding padding];
