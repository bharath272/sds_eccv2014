function [cls, inst, categories]=load_gt(sbd_dir, imid)
x=load(fullfile(sbd_dir, 'cls', [imid '.mat']));
cls=x.GTcls.Segmentation;
x=load(fullfile(sbd_dir, 'inst', [imid '.mat']));
inst=x.GTinst.Segmentation;
instbdry=x.GTinst.Boundaries;
categories=x.GTinst.Categories;

