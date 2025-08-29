##  Repository for thesis "dynamic hierarchical stochastic block models" 

# Background 

This code repository is part of a master research project.  The file thesis.pdf contains the 
report.  In short, this project defines a new class of dynamic networks, called dynamic hierarchical 
stochastic block models.  Fixed nodes are statically assigned to super-communities and dynamically 
assigned to sub-communities, and adjacency matrices are created by independent-edge process 
with the connection probability between two nodes decided by their super-/sub-community membership.  

The code base establishes a simulation and an estimation procedure, measurement and plotting 
functionalities, and notebooks demonstrating these on simulated and real world data.  We will give 
here a rough overview of the code structure and main functionality, especially on how the 
jupyter notebooks contribute.  

# Installation 

After cloning the github repository via 

git clone https://github.com/jcreutzig/thesis my_folder, 

create a new python environment with python >= 3.12, and run 

pip install -r requirements.txt 

I tested this with a fresh python install and was able to subsequently run all jupyter notebook code in VisualStudio.  

# Model simulation 

The modules dyn_dcsbm.py and dyn_hdscbm.py establish model simulations.  This is an incremenetal process, 
where in dyn_dcsbm.py we introduce the fundamental class dynamicRDPG, which uses a latent position 
generator object for the embedding Xt at time t, to calculate Pt = Xt Xt^T and simulate the adjacency matrix 
At.  We then implement Markov latent position generators for stochastic block models with individual random 
weights, which produce simple Markov random walks c(i, t) through the available community vectors, and return 
Xt with Xt_i = w(i,t) * c(i,t) with weights w(i,t) e [0,1] simulated as iid Beta distributed.   

In dyn_hdscb.py, we introduce custom convenience functionality for mapping subgraph indices to [0..n_k] 
index ranges, and back, in SubGRaphIndexMapper.  This is then used in HierarchicalMarkovDCSBMGenerator 
which combines static super-community assignment and sub-community assignment (from the Markov DCSBM generators 
mentioned above).  The class HierarchicalDynamicDCSBM is then a specialization of the dynamicRDPG class 
using this generator.  

The convenience function make_hierarchical_model is meant as the main user functionality for creating models.  
This utilizes a helper function generate_hierarchical_community_vectors that can create super-/sub-community vectors 
with prescribed dot products, and feeds them into a HierarchicalMArkovDCSBMGenerator.  The result is an 
HierarchicalDynamicDCSBM object/model which can be queried by .get_adjacency_matrices, to produce adjacency matrices 
as a list of numpy multiarrays.  

# Estimation and evaulation functions 

The file estimators.py contains several partial functionalities.  The methods are described in thesis.pdf as 
algorithms, and should have quite recognizable names.  The end user should relate mainly to the function 
dynamic_hierarchical_dcsbm_detection_stable which implements Algorithm 9 from thesis.pdf.  A secondary 
function is dynamic_hierarchical_dcsbm_detection_simple, which implemenmts Algorithm 7 and differs from 
dynamic_hierarchical_dcsbm_detection_stable in that it uses an Elbow cutoff based on singular values 
instead of spatio-temporal signal/noise ratio.  

The file ari_test.py contains useful tools for measurement.  Firstly it defines a class HierarchicalCommunityStructure, 
which provides a simple, unified interface for both estimated and ground truth community assignments, with the 
class method constructors .from_results and .from_model.  The function evaluate_from_hcs assumes two such objects 
to be provided, one built from the ground truth and one from estimation results, and calculates super ARI and mean 
ARI as detailed in thesis.pdf.  

# Simulating and estimation on simulated data 

The main tool for running simulation and estimation is the notebook ari_test_notebook.ipynb.  
This notebook is set up to run multiple test runs over each parameter setting.  Since a single run can 
take 1.5-3 minutes, with 10 runs per parameter setting and about 20 combinations to test, the run time of 
this notebook can be significant.  We provided pre-run simulations for convenience, in output/stable and 
output/simple.  To run fresh simulations, move or delete the .pkl files in these locations and run this notebook.  

The notebook first defines parameter defaults and ranges as per Table 1 in thesis.pdf, then a wrapper method 
ExperimentManager which handles running single experiments as well as loading and saving results.  The helper 
function run_parameter_sweep uses the ExperimentManager and a custom generator for parameter values, to 
simulate a model using make_hierarchical_model, and then estimates communities using 
dynamic_hierarchical_dcsbm_detection_stable.  It finally evaluates the performance of estimation using 
evaluate_from_hcs.  The method run_all_experiments accepts as input an estimation method and a target folder, 
as well as the number of repetitions to be done for each parameter configuration.  

Importantly, the code acknowledges that the user might have to stop and resume the estimation, and provides 
a simple support mechanism.  After each batch of test runs for a given parameter, the results are stored 
in a .pkl file, and when run, the algorithm skips over each parameter setting that already has data in the .pkl file.  
In order to preserve reproducibility, the method re-sets the random seed to a global random seed, plus a hash 
depending on the current parameter name and value, before each batch of test runs.  

The notebook plot_ari_results.ipynb consists of one large print function that reads the produced .pkl files 
and plots the results in a multi-plot.  

# Estimation on the TfL Cycle Infrastructure Dataset 

In the folder data/, two csv files with time-stamped journey data are found, as well as one file santander_locs.csv.  These 
files originate from https://cycling.data.tfl.gov.uk and are subject the Open Government Licence 2.0, and to the terms and 
conditions listed under https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service.  

The notebook process_TfL_CID.ipynb will, when run, read in these data and produce two outputs.  First, it generates 
one adjacency matrix per day, where on each day nodes i and j are connected iff there was at least one journey i -> j 
or j -> i on that day, and stores these in the file output/am_bikes.pkl.  Secondly, it maps the nodes (indices) of the 
adjacency matrix to latitude/longitude pairs as far as possible (not all stations could be mapped), and stores the result 
in data/node_list_extended.csv.  

The notebook bike_data_estimation.ipynb does the (fairly small) job of loading the adjacency matrices and applying 
dynamic_hierarchical_dcsbm_detection_stable to extract estiamted super-/sub community memberships.  The results are 
stored in bike_results.pkl.  

The notebook plot_bike_data will pick up this data and generate both a dashboard plot on the statistics of community 
membership, and two geospatial plots, saved under bike_dashboard.pdf, lat_lon_super_c_with_basemap.pdf and bs_latlon.pdf.  

