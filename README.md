# AffVids_mvpa

*Yiyu Wang 2022 October*


---
output: html_document
---
## Neural Predictors of Subjective Fear depend on the Situation
The code and analysis for our paper <i> Neural Predictors of Subjective Fear depend on the Situation </i> is hosted on GitHub at [https://github.com/yiyuwang/AffVids_mvpa](https://github.com/yiyuwang/AffVids_mvpa). In service of other researchers who would like to reproduce our work or are interested in applying the same models in their own work, we have made our data and code available. If you have any questions, or find any bugs (or broken links), please email me at wang.yiyu@northeastern.edu. A copy of a preprint is available [here](https://www.biorxiv.org/content/10.1101/2022.10.20.513114v1).

### Data
LASSOPCR+Searchlight results, permutation test results, and behavioral data can be downloaded here: 
  - [LASSOPCR searchlight results](results/searchlight_wholebrain/)  
  - [Permutation test results](results/permutation_test/)
  - [task data](BehavData/AffVids_novel_interpolated_rating_zscored.csv)


  
### Analysis Code
* [Behavioral data analysis](0_Preprocess_fear_ratings.ipynb)   
* [First Level GLM](1_Create_GLM_beta.ipynb)
* [LASSOPCR in searchlight](2_LASSOPCR_Searchlight.ipynb)   
* [Create Permutation distributions](scripts/Classification.ipynb)
* [Permutation testing](4_PermutationTest_organize)  
* [Results summary and visualization](5_Visualization.ipynb)

