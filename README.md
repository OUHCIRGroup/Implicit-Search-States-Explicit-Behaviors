## README

### Extracting the Implicit Search States from Explicit Behavioral Signals in Complex Search Tasks

This repository contains a consolidated script of a poster published in ASIST2021 for processing and visualizing data from three datasets: TianGong, KDD19, and track2014.

#### Requirements:

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scipy
- sklearn

#### Usage:

1. Ensure you have the datasets downloaded and placed in the appropriate directory. Adjust the file paths in the script if necessary.
2. Run the `data_process.py` script:

```bash
python data_process.py
```

3. The script will process each dataset and provide visualizations based on the pre-analysis logic.

#### Function Descriptions:

- `dataset_analysis(dataset_name)`: Processes the specified dataset and returns extracted data.

- `pre_analysis(dataset_name)`: Integrates the dataset analysis and contains clustering analysis and visualization logic for the specified dataset.

#### Datasets:

The datasets can be found at the following links:

- [TianGong Dataset](http://www.thuir.cn/tiangong-ss-fsd/)
- [KDD19 Dataset](http://www.thuir.cn/KDD19-UserStudyDataset/)
- [TREC14-Session track Dataset](https://trec.nist.gov/data/session2014.html)

#### Publication:

If you use this code, please cite our publication:

```latex
@article{https://doi.org/10.1002/pra2.587,
author = {Wang, Ben and Liu, Jiqun},
title = {Extracting the Implicit Search States from Explicit Behavioral Signals in Complex Search Tasks},
journal = {Proceedings of the Association for Information Science and Technology},
volume = {58},
number = {1},
pages = {854-856},
keywords = {behavioral features, clustering analysis, Complex search task, task state},
doi = {https://doi.org/10.1002/pra2.587},
url = {https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/pra2.587},
year = {2021}
}
```

#### Acknowledgment:

This work is supported by the Junior Faculty Fellowship Program award from the University of Oklahoma Office of the Vice President for Research and Partnerships.



For any issues or questions regarding the code, please [contact us](mailto:benw@ou.edu). 
