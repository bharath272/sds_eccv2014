function im_crop=clip_img(img, box)
box=round(box);
sz=size(img);
clipped_box=min(max(round(box), [1 1 1 1]),sz([2 1 2 1]));
diff=clipped_box-box;
im_clip=img(clipped_box(2):clipped_box(4), clipped_box(1):clipped_box(3),:);
im_crop=zeros(box(4)-box(2)+1, box(3)-box(1)+1,size(img,3));
im_crop(1+diff(2):diff(2)+size(im_clip,1),1+diff(1):diff(1)+size(im_clip,2),:)=im_clip;

