# tracET: A Software for tracing low-level structures in cryo-Electron Tomography


## Requirements
* Python with the packages listed in the document [requirements.txt](https://github.com/PelayoAlvarezBrecht/tracer/tree/pypi/requirements.txt)

## Instalation
You can install tracET in two ways:
* From PyPI:
  * In a python terminal, write *pip install https://test.pypi.org/simple/ tracET*
  * Look for the scripts in the path *\YOUR_PATH\anaconda3\Lib\tracET\scripts*
* From Github:
  * Clone the repository https://github.com/PelayoAlvarezBrecht/tracer/
  * In a terminal, open the subdirectory *cmodules*
  * Execute the command *python setup.py install*


## Scripts
There is five different scripts to apply different parts of the process:

### Saliency map:
* Script description:
  * The name of the script is get_saliency.py
  * From a tomogram with a binary segmentation, calculates the saliency map, a distance transfromation of the input softed with a gaussian filter.
  * This step is also included in the apply_nonmaxsup.py script.

* Parameters:
  * The parameter *in_tomo*, called with "-i" or "--itomo", needs the name of the input file. A binary map tomogram in a mrc format or nrrd format.
  * The parameter *soomth_desviation*, called with "-s" or "-sdesv", is the desviation for gaussian filter and it should be ~1/3 of the element radium.

* Outputs:
  * A tomogram with the saliency map, in the same format of the input and with the same name with the sufix _*saliency*

### Non-Maximum Suppression:
* Script description:
  * The name of the script is apply_nonmaxsup.py
  * From a segmentation or a saliency map, detect the most central voxels of the elements and construct an equiespatial point cloud of this elements. 
  

* Parameters:
  * The parameter *in_tomo*, called with "-i" or "--itomo", needs the name of the input file. A scalar or binary map tomogram in a mrc format or nrrd format.
  * The parameter *soomth_desviation*, called with "-s" or "-sdesv", is the desviation for gaussian filter and it should be ~1/3 of the element radium.
  * The parameter *skel_mode*, called with "-m" or "--mode", is the structural mode for computing the skeleton: "s" for surfaces, "l" for lines and "b" for blobs.
  * The parameter *binary_input*, called with "-b" or "--ibin", needs to be 0 if is a scalar map, and 1 if is a binary map. In this case, it calculates the distance transformation saliency map.
  * The parameter *filter*, called with "-f" or "--filt", is the filter for the mask of the suppression. Is optional and if is not given, only eliminate negative values.
  * The parameter *downsample*, called with "-d" or "--downs", if is given, apply a downsample of the radius indicated.
  

* Outputs:
  * A tomogram with only the maximums of the saliency map, in the same format of the input and with the same name with the sufix _*supred*

### Spatial Embedded Graph
* Script description:
  * The name of the script is trace_graph.py
  * From a point cloud of filaments, trace a spatial emmbeded graph, calculates the different connect components, the different subbranches and model every branch as a curve to meassure the different properties.
  

* Parameters:
  * The parameter *input*, called with "-i" or "--itomo", is the tomogram with the point cloud of the filament segmentation, in mrc or nrrd format. (The output of the previous script).
  * The parameter *radius*, called with "-r" or "--rad," is the radius for connect points in the graph.
  * The parameter *subsampling*, called with "-s" or "--subsam", is the radius used for subsample points. If is not given, there is not subsampling.
  

* Outputs:
  * A vtp with the information of the graph components, branches and geometric data, with the same name of the input and the extension "_skel_graph.vtp"
  * A csv with the information of the graph components, branches and geometric data, with the same name of the input and the extension "_skel_graph.csv"

### Blobs Clustering
* Script description:
  * The name of the script is Get_cluster.py
  * From a point cloud tomogram of blobs, cluster the points using MeanShift or Affinity Propagation, and localize the centroids.

* Parameters:
  * The parameter *input*, called with "-i" or "--itomo", is the tomogram with the point cloud of the filament segmentation, in mrc or nrrd format. (The output of the previous script).
  * The parameter *mode*, called with "-m" or "--mode", is the parameter to select the algorithm of clustering:
    * If is "Affinity", Affinity propagation is used. Is only recomended for small tomograms.
    * If is "MeanShift", Mean Shift algorithm is used. This is recomended for all type of tomograms, but need two parameters more:
      * The parameter *blob_diameter*, called with "-b" or "--blob_d", is the diameter of the blobs planned to detect.
      * The parameter *n_jobs*, called with "-n" or "--n_jobs", is the number of jobs to execute the algorithm in paralelle.

* Outputs:
  * A vtp file with the points of the ribosomes labeled with the clusters they are part, with the same name of the input and the extension "*mode*_labeled.vtp".
  * A mrc file with the points of the ribosomes labeled with the clusters they are part, with the same name of the input and the extension "*mode*_labeled.mrc".
  * A txt file, convertible to IMOD .mod file, with the information of the centroid of every cluster.

### Membrane clasification
* Script description:
  * The name of the script is membrane_poly.py
  * From a point cloud of membranes, it cluster the points in the diferent membranes.

* Parameters:
  * The parameter *in_tomo*, called with "-i" or "--itomo", is the tomogram with the point cloud of the membrane segmentation, in mrc or nrrd format. (The output of the previous script).
  * The parameter *distance_clustering*, called with "-d" or "--dist", is the distance of points to be part of the same cluster.
  * The parameter *min_samples*, called with "-s" or "--samp", is the minimum samples needed to make a cluster. Is optional and if is not given, it takes value 2.

* Outputs:
  * A vtp file with the points of the membranes labeled with the clusters (diferent membranes) they are part, with the same name of the input and the extension ".vtp".

### DICE METRIC

* Script description:
  * The name of the script is seg_skel_dice.py
  * From two different binary segmentations, it calculates the TS, TP and DICE metric, and give the two skeletons of the inputs.

* Parameters:
  * the parameter *in_tomo*, called with "-i" or "--itomo", a tomogram with a binary segmentation, in mrc or nrrd format.
  * The parameter *gt_tomo*, called with "-g" or "--igt" the ground truth segmentation, in mrc or nrrd format.
  * The parameter *skel_mode*, called with "-m" or "--mode", is the structural mode for computing the skeleton: "s" for surfaces, "l" for lines and "b" for blobs.

* Outputs:
  * TS metric
  * TP metric
  * DICE metric
  * (Optional) Asked with "-o" or "-otomo", skeleton of the input tomogram.
  * (Optional) Asked with "-t" or "-ogt", skeleton of the ground truth tomogram.

