function im1=color_seg(seg, img, color,ucm)
img=im2double(img);
im1=zeros(size(img));
img=rgb2gray(img);
if(~exist('color', 'var'))
	color=[0 1 1];
end
im1(:,:,1)=0.5*color(1)*seg+0.5*img;
im1(:,:,2)=0.5*color(2)*seg+0.5*img;
im1(:,:,3)=0.5*color(3)*seg+0.5*img;
if(nargin>3)
stren=ucm.strength(3:2:end, 3:2:end);
stren=(stren<0.1);
im1=bsxfun(@times, im1, double(stren));
end
