# Data Quality Index (DQI)

As per some recent study there have been some serious revelations in the field of machine learning. When Neural Model achieved human level performance across various natural language tasks without having full access to proper knowledge, led humans to question how the neural models are actually solving these datasets. 

Recently, a series of works Gururangan et al 2018, Poliak et al 2018, Kaushik et al 2018, Tsuchiya et al 2018, Tan et al 2019 , Schwartz et al 2017 has shown that many of the popular datasets, such as SQUAD Rajpurkar et al 2016 and SNLI Bowman et al 2015 have unwanted biases Torralba et al 2011 resulting from the annotation process. The spurious biases represent "unintended correlations between input and output" as mentioned by Ronan et al Bras et al 2020. Models exploit these biases as features instead of utilizing the actual underlying features needed to solve a task. 

The DQI aims at 
  * finding the parameters that causes such biases in the existing datasets, and
  * curb these biases at their root so that better datasets can be created in the future.
