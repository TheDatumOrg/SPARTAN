# Artifact and Reproducibility Submissions

## Platform
+ 2xAMD EPYC 7713 64-Core processors 
+ Ubuntu 22.04.3 LTS operating system

## Preparation

To install SPARTAN you will need to follow

**Step 1**: Install the required dependencies

```shell
# create your environment, and then
pip install -r requirements.txt
```

**Step 2**: Preparing the datasets for time-series analytical tasks under `data` folder:

+ Download [the UCR Archive](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip) for classification, clustering, and the tightness of lower bound (TLB) tasks.
+ Download the [TSB-UAD Archive](https://www.thedatum.org/datasets/TSB-UAD-Public.zip) for time-series anomaly detection.
+ Download the [synthetic CBF dataset](https://drive.google.com/drive/folders/1enRCCpJrtHRZkYSsbq_mZEFro3hk9evX?usp=sharing) for scaling experiments.


## Experiment Reproduction
In this artifact and reproducibility submission, we include all four core components from the paper: classification, clustering, tightness of lower bound (TLB), and anomaly detection, as well as ablation studies. Feel free to contact the author by yang.7007@osu.edu if there're any questions and we are more than happy to help.

### Section 5.1 Benchmark on Baseline (classification)

Run experiments:

```
# enter the project directory
# accuracy
bash scripts/Section5_1_run_classification_benchmark.sh
# runtime (approximation stage)
python scripts/Section5_1_run_runtime_benchmark.py
```

Draw Plots:
```
python scripts/Section5_1_draw_CD.py
python scripts/Section5_1_draw_scatter.py
```

### Section 5.2 Classification 

Run experiments:
```
bash scripts/Section5_2_run_classification.sh
```

Draw Plots:
```
python scripts/Section5_2_draw_CD.py
python scripts/Section5_2_draw_pairwise.py
```

Run ablation experiments:
```
bash scripts/Section5_2_run_daa_stategy_ablation.sh
bash scripts/Section5_2_run_daa_param_ablation.sh
bash scripts/Section5_2_run_binning_ablation.sh
```

Draw ablation plots:
```
python scripts/Section5_2_draw_CD_daa_strategy_ablation.py
python scripts/Section5_2_draw_plot_daa_param_ablation.py
python scripts/Section5_2_draw_pairwise_binning_ablation.py
```

### Section 5.3 TLB 

Run experiments:
```
bash scripts/Section5_3_run_tlb.sh
python scripts/Section5_3_run_tlb_aggregation_result.py
```

Draw Plots:
```
python scripts/Section5_3_draw_3dbar.py
python scripts/Section5_3_draw_CD.py
python scripts/Section5_3_draw_pairwise.py
```

### Section 5.4 Clustering

Run experiments:
```
bash scripts/Section5_4_run_clustering.sh
```

Draw Plots:
```
python scripts/Section5_4_draw_CD.py
python scripts/Section5_4_draw_pairwise.py
```

### Section 5.5 Anomaly Detection

Run experiments:
```
bash scripts/Section5_5_run_anomaly.sh
```

Plot:
```
python scripts/Section5_5_draw_anomaly_CD.py
```

### Section 5.6 Accuracy-to-runtime Analysis

Run experiments:
```
#SPARTAN runtime analysis
python scripts/Section5_6_spartan_runtime_analysis.py
#SPARTAN family on UCR
bash scripts/Section5_6_run_spartan_family.sh
#Scaling experiments
bash scripts/Section5_6_run_scaling_acc.sh
bash scripts/Section5_6_run_scaling_time.sh
```

Draw Plots:

```
python scripts/Section5_6_draw_CD.py
python scripts/Section5_6_draw_spartan_bar_plot.py
python scripts/Section5_6_draw_plot_varylen_acc_update.py
python scripts/Section5_6_draw_plot_varylen_time_update.py
python scripts/Section5_6_draw_plot_varynum_acc_update.py
python scripts/Section5_6_draw_plot_varynum_time_update.py
```

### Section 5.7 Revisiting the Symbolic Distance Measure

Run experiments
```
bash scripts/Section5_7_run_existing_dist.sh
```

Draw Plots:
```
python scripts/Section5_7_draw_pairwise.py
python scripts/Section5_7_draw_CD.py
```

### Section 5.8 Comparing with Euclidean Distance

Run experiments
```
bash scripts/Section5_8_run_classification_alpha.sh
bash scripts/Section5_8_run_classification_wordlen.sh
bash scripts/Section5_8_run_clustering_alpha.sh
bash scripts/Section5_8_run_clustering_wordlen.sh
bash scripts/Section5_8_run_euclidean.sh
```

Draw Plots
```
python scripts/Section5_8_draw_classifcation_varying_alpha.py
python scripts/Section5_8_draw_classifcation_varying_wordlen.py
python scripts/Section5_8_draw_clustering_varying_alpha.py
python scripts/Section5_8_draw_clustering_varying_wordlen.py
python scripts/Section5_8_draw_anomaly_CD.py
```








