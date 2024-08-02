#!/bin/bash

#SBATCH --job-name=lassopcr_sl                  # sets the job name
#SBATCH --nodes=1                                 # reserves 1 machines
#SBATCH --cpus-per-task=16                         # sets 8 core for each task
#SBATCH --mem=100Gb                               # reserves 100 GB memory
#SBATCH --partition=short                 	  	  # requests that the job is executed in partition short
#SBATCH --time=23:59:00                           # reserves machines/cores for 5 days = 120 hrs.
#SBATCH --output=output_permutation/7permutation_lassopcr_sl.%A-%a.out         # sets the standard output to be stored in file %A-%a - Array job id (A) and task id (a))
#SBATCH --error=output_permutation/7permutation_lassopcr_sl.%A-%a.err          # sets the standard output to be stored in file %A-%a - Array job id (A) and task id (a))

module load python/3.7.0

cd /work/abslab/Yiyu/AffVids/Code/LASSOPCR_searchlight/SL_LASSOPCR_001_subjective_visreg_fixedcv_interpolate/

srun python SUPPLEMENTARY_10_StimConstant.py '2'