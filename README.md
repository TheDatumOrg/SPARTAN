# SPARTAN: SPARTAN: Data-adaptive Symbolic Representations for Time Series Data Analysis

---

## Abstract
Time series data mining is an important area receiving increasing attention with applications to diverse disciplines and industries. Critical tasks of time series analysis, such as similarity search, indexing, classification, and anomaly detection, rely on methods that reduce the dimensionality of input time series via a transformation to a string of discrete symbols, called \textit{Symbolic Representation}. Despite decades of progress in this area, the majority of solutions still rely on a well-established method for its simplicity, which has been proposed for two decades. Unfortunately, this family of solutions are far from optimal, as they fail to capture the disproportionate importance of subspaces in alphabet dictionary creation, which leads to a loss of information. Therefore, to address these issues, we propose \textbf{SPARTAN}, a novel \textit{data-adaptive} method that encodes the data by intelligently adapting the alphabet dictionary across dimensions to mitigate the existing limitations. In general, SPARTAN exploits intrinsic dimensionality reduction properties to derive the orthogonal subspaces for representation. Based on this, SPARTAN solves a constrained optimization problem using dynamic programming to allocate the alphabet size with an awareness of the importance of each subspace. To demonstrate the
robustness of SPARTAN, we first perform an extensive evaluation of 7 state-of-the-art methods over 128 UCR datasets, the largest study today in this area. SPARTAN significantly outperforms the current methods in all dimensions, achieving $2\times$ better tightness of lower bounding and statistically significant improvements in classification downstream tasks. Importantly, SPARTAN also improves by $2\times$ inference runtime performance compared to the current best solution. Overall, this innovative approach offers a comprehensive solution for time series analysis with a better balance between the representation ability and concerns of time and memory efficiency.

## Getting Started

To install SPARTAN you will need the following tools
- git
- pip

**Step 1**: Clone this repository and change directory to the home directory
bash`
git clone https://github.com/TheDatumOrg/SPARTAN
cd SPARTAN
`

**Step 2**: Install the required dependencies
bash`
pip install requirements.txt
`

**Step 3**: Install UCR Archive Data
bash`
wget http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip
unzip Univariate2018_ts.zip
`

## Evaluation

To test SPARTAN classification accuracy on a single dataset:

bash`
python3 train.py --data /path/to/your/data --problem DatasetName --classifier spartan --config ./path/to/model_params/
`

To test SPARTAN accuracy on all 128 datasets:
bash`
python3 experimental_evaluation.py --data /path/to/your/data --problem DatasetName --classifier spartan --config ./path/to/model_params/
`
