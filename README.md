##Simultaneous Detection and Segmentation

This is code for the ECCV Paper:  
[Simultaneous Detection and Segmentation](http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathECCV2014.pdf)  
Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik  
To appear in ECCV, 2014.  


###Installation


* **Installing caffe:**
  The code comes bundled with a version of caffe that we have modified slightly for SDS. (These
  modifications might be merged into the public caffe version sometime in the future). To install
  caffe, follow the instructions on the [caffe webpage](caffe.berkeleyvision.org). (You'll have to
  install some pre-requisites). After installing all prerequisites, cd into `extern/caffe` and do `make caffe`.  
  After you have made caffe, you will also need to do `make matcaffe`.

* **Downloading other external dependencies (MCG and liblinear):**
  The extern folder has a script that downloads MCG and liblinear and compiles liblinear. 
  After running the script, cd into `extern/MCG-PreTrained` and change the path in `root_dir.m` to the path to the MCG-PreTrained
  directory.

* **Starting MATLAB:**
  Start MATLAB and call `startup_sds` from the main SDS directory. This will compile all
  mexes in MCG and liblinear, and add all paths.

  A few possible issues related to Caffe:
  + You may need to add the path to CUDA libraries (usually in /usr/local/cuda/lib64)
    to `LD_LIBRARY_PATH` before starting MATLAB.
  + When running the code, if you get an error saying:
    `/usr/lib/x86_64-linux-gnu/libharfbuzz.so.0: undefined symbol: FT_Face_GetCharVariantIndex`,
    try adding `/usr/lib/x86_64-linux-gnu/libfreetype.so.6`(or the equivalent library that
    your system may have) to the `LD_PRELOAD` environment variable before starting MATLAB. 
 



###Using Pre-computed results
To get started you can look at precomputed results.
Download the precomputed results from this ftp link:
`ftp://ftp.cs.berkeley.edu/pub/projects/vision/sds_precomputed_results.tar.gz`
and untar it. The precomputed results contain results on VOC2012 val images (SDS, detection and segmentation). 
You can visualize the precomputed results using the function `visualize_precomputed_results.m`:
`visualize_precomputed_results('/path/to/precomputed/results', '/path/to/VOC2012/VOCdevkit/VOC2012/JPEGImages', categ_id)`;   
Here `categ_id` is the number of the category, for example 15 for person.

Note that you **do not** need to install Caffe or any of the external dependencies above if you want to simply visualize
or use precomputed results.

###Testing Pre-trained models

Download the pretrained models from this ftp link:
`ftp://ftp.cs.berkeley.edu/pub/projects/vision/sds_pretrained_models.tar.gz`
and untar them in the main SDS directory. 

`demo_sds.m` is a simple demo that uses the precomputed models to show the outputs we get on a single image. It takes no arguments.
It runs the trained models on an example image and displays the detections for the person category.
This function is a wrapper around the main function, which is called `imagelist_to_sds.m`.

###Benchmarking and evaluation

You can also run the benchmark demo, `demo_sds_benchmark`, which tests our pipeline on a small 100 image subset of
VOC2012 val and then evaluates for the person category. You can call it as follows:  
`demo_sds_benchmark('/path/to/VOC2012/VOCdevkit/VOC2012/JPEGImages/', '/path/to/cachedir', '/path/to/SBD');`  
Here the cachedir is a directory where intermediate results will be stored. The function also requires the SBD
(Semantic Boundaries Dataset), which you can get [here](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html).
The function does the evaluation for both before refinement and after refinement, and reports an AP<sup>r</sup> of **59.9** in the first case and **66.8** in the second case. 



The main function for running the benchmark is `evaluation/run_benchmark.m`. `demo_sds_benchmark` should point 
you to how to run the benchmark.

###Evaluating on detection and segmentation

* **Detection:**
  Look at `imagelist_to_det.m` to see how to produce a bounding box detection output. 
  In summary, after computing scores on all regions, we use `misc/box_nms.m` to non-max suppress the boxes
  using box overlap. `misc/write_test_boxes` then writes the boxes out to a file that you can submit to PASCAL.

* **Semantic segmentation:**
  Look at `imagelist_to_seg.m` to see how we produce a semantic segmentation output.
  In summary, after we compute scores on all regions, we do `misc/region_nms.m` to non-max suppress boxes,
  and use `misc/get_top_regions.m` to get the top regions per category. For our experiments, we picked the top 5K regions for seg val
  and seg test. Then we call `paste_segments`:
  `[local_ids, labels, scores2] = paste_segments(topchosen, scores, region_meta_info, 2, 10, -1);`
  `topchosen` is the first output of `get_top_regions.m`. These parameters above were tuned on seg val 2011.
  This function will pick out the segments to paste. To do the actual pasting, use `create_pasted_segmentations` (if you don't want any
  refinement) or `create_pasted_segmentations_refined` (if you want refinement). Refinement is a bit slower but works ~1 point better.



###SDS results format
If you want to do more with our results, you may want to understand how we represent our results.
* **Representing region candidates:**
 Because we work with close to 2000 region candidates, saving them as full image-sized masks
 uses up a lot of space and requires a lot of memory to process. Instead, we save these region
 candidates using a superpixel representation: we save a superpixel map, containing the superpixel id 
 for each pixel in the image, and we represent each region as a binary vector indicating which
 superpixels are present in the region. To allow this superpixel representation to be accessible to
 Caffe, we 
 + save the superpixel map as a text file, the first two numbers in which represent the size of the
 image and the rest of the file contains the superpixel ids of the pixels in MATLAB's column-major order
 (i.e, we first store the superpixel ids of the first column, then the second column and so on).
 + stack the representation of each region as a matrix (each column representing a region) and save it as a png image.

 `read_sprep` can read this representation into matlab.

* **Representing detections:**
 After the regions have been scored and non-max suppressed, we store the chosen regions as a cell array, one cell
 per category. Each cell is itself a cell array, with as many cells as there are images, and each cell containing
 the region id of the chosen regions. The scores are stored in a separate cell array.

* **Representing refined detections:**
 After refinement, the refined regions are stored as binary matrices in mat files, one for each image. The refined
 regions for different categories are stored in different directories




###Retraining region classifiers

To retrain region classifiers, you first need to save features for all regions including ground truth. You can look at the function
`setup_svm_training.m`. This function will save features and return a `region_meta_info` struct, which has in it the overlaps of all the
regions with all the ground truth. The function expects a list of images, a number of paths to save stuff in, and a path to the
ground truth (SBD).

Once the features are saved you can use the `region_classification/train_svms.m` function to train the detectors.
You can also train refinement models for each category using `refinement/train_refiner.m` 

###Retraining the network
To retrain the network you will have to use caffe. You need two things: a prototxt specifying the architecture, and a window file specifying
the data.

* **Window file:**
Writing the window file requires you to make a choice between using box overlap to define ground truth, or using region overlap to define ground
truth. In the former case, use `feature_extractor/make_window_file_box.m` and in the latter use `feature_extractor/make_window_file_box.m`. Both functions
require as input the image list, `region_meta_info` (output of `preprocessing/preprocess_mcg_candidates`; check `setup_svm_training` to see how to call it), 
sptextdir, regspimgdir (specifying the superpixels and regions) and the filename in which the output should go.

* **Prototxt:**
There are 3 prototxts that figure during training. One specifies the solver, and points to the other two: one for training and the other for testing.
Training a single pathway network for boxes can be done with the `window_train` and `window_val`, a single pathway network on regions can be done using `masked_window_train`
and `masked_window_val`, and a two pathway network (net C) can be trained using `piwindow_train` and `piwindow_val`. (Here "pi" refers to the architecture of the network,
which looks like the capital greek pi.)
The train and val prototxts also specify which window file to use.
The solver prototxt specifies the path to the train and val prototxts. It also specifies where the snapshots are saved. Make sure that path can be saved to.

* **Initialization:**
 A final requirement for finetuning is to have an initial network, and also the imagenet mean. The latter you can get by running 
 `extern/caffe/data/ilsvrc12/get_ilsvrc_aux.sh`
 The initial network is the B network for net C. For everything else, it is the caffe reference imagenet model, which you can get by running
 `extern/caffe/examples/imagenet/get_caffe_reference_imagenet_model.sh`
 
* **Finetuning:**
 cd into caffe and use the following command to train the network (replace `caffe_reference_imagenet_model` by the appropriate initialization):  
 `GLOG_logtostderr=1 ./build/tools/finetune_net.bin ../prototxts/pascal_finetune_solver.prototxt ./examples/imagenet/caffe_reference_imagenet_model 2>&1 | tee logdir/log.txt`  
 Finally, extracting features requires a network with the two-pathway architecture. If you trained the box and region pathway separately, you can stitch them together
 using `feature_extractor/combine_box_region_nets.m`




