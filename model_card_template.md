# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The person developing this model is David Wang. 

The model date is November 25, 2024. The model version is 1.0. 

The model type is random forest classifier. 

The hyperparameters include random state and n_jobs, which is 42 and -1 respectively. 

The categorical features are workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Intended Use

This model is intended to classify individuals based on their income level using demographic and work-related attributes. 

It can be used for social research on income disparities, market segmentation, and targeted surveys for individuals earning above or below a threshold income.

An out of scope example would be automated decision-making processes, such as loan approvals, without proper fairness audits.

## Training Data

The dataset source is the census.csv that is provided by UCI in their machine learning repository. 

The dataset contains demographic and economic information for individuals, including features such as work class, education, marital status, and occupation. 

The label is the salary column, indicating whether income is  greater than $50K or less than or equal to $50K. 

Finally, the data was split 80% for training and 20% for testing.


## Evaluation Data

The evaluation data is the 20% of the original dataset held out during the train_test_split.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

The model’s performance metrics were computed using the compute_model_metrics function. It evaluates how well the model predicts income levels on the test set.

## Ethical Considerations

The dataset contains features such as race, sex, and native-country, which may introduce bias into the model. 

The dataset might contain sensitive information, so we should strive to ensure data handling complies with any privacy laws.

## Caveats and Recommendations

Random Forest models are less interpretable than simpler linear models, so feature importance should be evaluated.

The model is trained on a specific census dataset. Therefore, its performance might not generalize to other populations or datasets.. 