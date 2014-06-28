function newimg=place_indices(bbox, Wsz, sz)

%% newimg = place_indices(bbox, Wsz, sz)
%% marks indices in the image where the different cells of W would land
%% bbox is [xmin ymin xmax ymax]
%% Wsz is size(W);
%% sz is size(img)


newimg=zeros(sz(1),sz(2));

r_ind=1:sz(1);
c_ind=1:sz(2);


W=[1:prod(Wsz)]; W=reshape(W, Wsz);
       
%Divide the detection window into bins along x and y
t_i=linspace(bbox(2)-1, bbox(4),Wsz(1)+1);
t_j=linspace(bbox(1)-1, bbox(3),Wsz(2)+1);
        
%bin the pixels
[ni, bini]=histc(r_ind,t_i);
[nj, binj]=histc(c_ind,t_j);
bini(r_ind==bbox(2)-1)=0;
binj(c_ind==bbox(1)-1)=0;
bini(bini==Wsz(1)+1)=Wsz(1);
binj(binj==Wsz(2)+1)=Wsz(2);
      

%find the pixels that lie inside the window
indi = find(bini>=1 & bini<=Wsz(1)); indj=find(binj>=1 & binj<=Wsz(2));
%multiply by appropriate weights and add
		
newimg(r_ind(indi), c_ind(indj)) = W(bini(indi),binj(indj));
