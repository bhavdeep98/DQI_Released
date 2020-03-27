# Data Quality Index (DQI)

Neural language models have achieved human level performance across several NLP datasets. However, recent studies have shown that these models are not truly learning how to perform the desired task; rather, their high performance is attributed to overfitting using spurious biases. In order to help dataset creators create datasets free of such unwanted biases, and dataset solvers adopt special methods that exploit the same, we introduce an empirical formula for Data Quality Index (DQI). We further tune this formula through rigorous experimentation. We have also proposed a novel adversarial filtering algorithm, Robust AFLite (RAFLite), to remove dataset biases. We show the efficacy of our approach across various NLI, Question Answering and Reading Comprehension datasets. Our work takes forward the process of dynamic dataset creation wherein datasets evolve together with the evolving state of the art, therefore serving as a means of benchmarking the true progress of AI. 

**Repository Structure**:\
**AFLite Implementation Python Notebook**: Our implementation of AFLite mentioned in Winogrande paper.\
**RAFLite**: Proposed Adversarial Filtering Algorithm implementation.\
**Viz**: UI work to incorporate DQI for providing feedback to data creators.\
**Papers**: List of papers read for identifying paramters along with their summaries in the excel sheet.\
**Pre Analysis Images**: Plots showing quality of various datasets we consider.

