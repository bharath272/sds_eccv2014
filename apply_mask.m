function reg2sp = apply_mask(pred, box, sp, Wsz)
newimg=place_indices(box, Wsz, size(sp));
newimg(newimg~=0)=pred(newimg(newimg~=0));
reg2sp=accumarray(sp(:), newimg(:))./accumarray(sp(:),1);




  
