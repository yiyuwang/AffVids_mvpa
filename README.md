# AffVids_mvpa

*Yiyu Wang 2024 November*

---
## Neural Predictors of Subjective Fear depend on the Situation
This is the repo for hosting the analysis code for our paper <i> Neural Predictors of Fear depend on the Situation </i> published at Journal Of Neuroscience. If you have any questions, or find any bugs (or broken links), please email me at yiyuwang@stanford.edu. A copy of a preprint is available [here](https://www.biorxiv.org/content/10.1101/2022.10.20.513114v1).

### Data
LASSOPCR+Searchlight results, permutation test results, and behavioral data can be downloaded here: 
  - [LASSOPCR searchlight results](results/searchlight_wholebrain/)  
  - [Permutation test results](results/permutation_test/)
  - [task data](BehavData/AffVids_novel_interpolated_rating_zscored.csv)


### Analysis Code
* [Behavioral data analysis](0_Preprocess_fear_ratings.ipynb)   
* [First Level GLM](1_Create_GLM_beta.ipynb)
* [LASSOPCR in searchlight](2_LASSOPCR_Searchlight.ipynb)   
* [Create Permutation distributions](3_Permutation.ipynb)
* [Permutation testing](4_PermutationTest_organize.ipynb)  
* [Results summary and visualization](5_Visualization.ipynb)
* [Supplementary Analysis: Stimulus Constant Analysis 10-11](SUPPLEMENTARY_11_Organize_Visualize_StimConstant.ipynb)




