#### The main body of the work is available in pdf format in the Assignment 3 folder at this repository link: https://github.com/Anton21a/Machine-Learning.git

### Data Cleaning and Pre-processing  

With the given specific of the assignment, the initial dataset was filtered to include only firms that were active in either 2012 or 2014. Thus, as an indicator of fast growing, the study employs the difference in value of the target variable between 2014 and 2012.

Firm age was computed based on the difference between the year and the founding year. Any negative values were set to zero. To get rid of outliers in target variable 'sales', its values were filtered by 10 million as upper-bandwidth limit and 1,000 as lower-bandwidth limit.

Next, the industry classification (ind2) was recoded into broader categories under a new variable called ind2_cat. This involved grouping various codes into categories such as 20, 30, 40, and 60 based on NACE industry code ranking. Firms with unclassified or missing industry codes were excluded from further analysis. A binary variable indicating whether the firm belongs to manufacturing or service type of activity was then created for the subsequent classification task. 

The engineering work was undertaken for variables which show lines from BS and P&L statements. A list of variables that should not be negative, such as sales, curr_assets, fixed_assets, inventories, etc. was used to filter out invalid observations, and these variables were subsequently transformed using the natural log (log1p()). The rest part of growth rate variables (such as profit or equity) were defined as a simple difference between values in 2014 and 2012, due to they could contain negative values. 

Growth rates and changes in KPI were then analyzed. Dummy variables were created to capture positive growth, extreme increases (over 100%), and severe declines (greater than 50% drop) for each variable. These dummy variables were encoded as categorical factors with "yes" and "no" values. Filters were applied to remove extreme values in the change of profit, tax, and equity indicators to avoid outliers that could skew modeling.

**Descriptive Statistics**

| Variable                  | Unique (#) | Missing (%) | Mean       | SD         | Min     | Median     | Max        |
|---------------------------|------------|--------------|------------|------------|---------|-------------|------------|
| comp_id                  | 16184      | 0            | 153708039710.4 | 138035363501.3 | 1001541.0 | 114609278976.0 | 464105013248.0 |
| sales                    | 13892      | 0            | 301218.1   | 872978.2   | 1000.0  | 61942.6     | 9963926.0  |
| founded_year             | 32         | 0            | 2002.7     | 7.0        | 1951.0  | 2003.0      | 2014.0     |
| ceo_count                | 7          | 0            | 1.3        | 0.5        | 1.0     | 1.0         | 7.0        |
| foreign                  | 10         | 0            | 0.1        | 0.3        | 0.0     | 0.0         | 1.0        |
| female                   | 10         | 0            | 0.3        | 0.4        | 0.0     | 0.0         | 1.0        |
| urban_m                  | 3          | 0            | 2.1        | 0.8        | 1.0     | 2.0         | 3.0        |
| age                      | 32         | 0            | 11.3       | 7.0        | 0.0     | 11.0        | 63.0       |
| new                      | 2          | 0            | 0.0        | 0.1        | 0.0     | 0.0         | 1.0        |
| ind2_cat                 | 12         | 0            | 47.1       | 12.6       | 20.0    | 56.0        | 60.0       |
| age2                     | 32         | 0            | 177.9      | 181.0      | 0.0     | 121.0       | 3969.0     |
| flag_asset_problem       | 2          | 0            | 0.0        | 0.0        | 0.0     | 0.0         | 1.0        |
| sales_2012               | 13625      | 0            | 251101.8   | 735154.4   | 1000.0  | 54022.2     | 9786907.0  |
| growth_ln_sales          | 16100      | 0            | 0.1        | 0.9        | −6.3    | 0.1         | 8.5        |
| growth_ln_curr_assets    | 16157      | 0            | 0.1        | 1.2        | −10.1   | 0.1         | 11.4       |
| growth_ln_curr_liab      | 16086      | 0            | 0.2        | 1.7        | −14.3   | 0.1         | 13.2       |
| growth_ln_extra_inc      | 1912       | 0            | 0.6        | 2.7        | −15.0   | 0.0         | 14.4       |
| growth_ln_fixed_assets   | 13595      | 0            | 0.6        | 2.7        | −15.6   | 0.0         | 13.3       |
| growth_ln_intang_assets  | 2547       | 0            | 0.0        | 1.9        | −12.3   | 0.0         | 13.9       |
| growth_ln_inventories    | 11103      | 0            | −0.1       | 3.1        | −14.6   | 0.0         | 13.5       |
| growth_ln_liq_assets     | 15959      | 0            | 0.2        | 1.9        | −13.1   | 0.2         | 13.4       |
| growth_ln_material_exp   | 16161      | 0            | 0.1        | 1.1        | −12.8   | 0.1         | 10.3       |
| growth_ln_personnel_exp  | 15031      | 0            | 0.0        | 2.2        | −12.7   | 0.1         | 13.9       |
| growth_ln_subscribed_cap | 504        | 0            | 0.1        | 0.5        | −9.3    | 0.0         | 8.0        |
| growth_ln_tang_assets    | 13457      | 0            | 0.7        | 2.8        | −15.6   | 0.0         | 13.3       |
| growth_ln_total_assets_bs| 16171      | 0            | 0.1        | 1.0        | −10.1   | 0.1         | 6.5        |
| growth_ln_amort          | 13563      | 0            | 0.3        | 2.3        | −10.7   | 0.0         | 12.3       |
| growth_ln_extra_exp      | 1390       | 0            | 0.1        | 2.1        | −13.4   | 0.0         | 13.3       |
| diff_extra_profit_loss   | 4685       | 0            | 0.0        | 0.5        | −27.7   | 0.0         | 34.5       |
| diff_inc_bef_tax         | 16163      | 0            | 0.4        | 4.1        | −91.1   | 0.0         | 126.8      |
| diff_profit_loss_year    | 15919      | 0            | 0.4        | 3.9        | −91.3   | 0.0         | 126.5      |
| diff_share_eq            | 16001      | 0            | −0.2       | 5.0        | −94.9   | 0.0         | 97.7       |
| growth_rate              | 16100      | 0            | 1.8        | 43.9       | −1.0    | 0.1         | 5091.1     |
| fast_growth              | 2          | 0            | 0.1        | 0.3        | 0.0     | 0.0         | 1.0        |


### Modelling
Each model employs 5 fold cross-validation method, so the dataset was split into training and holdout sets using an 80/20 ratio. The study defines different list of variable which were then embedded in logit model's specifications. The most basic specification (X1) includes firm characteristics such as age and categorical variables. Subsequent models (X2–X5) add log-transformed growth variables, binary indicators for high growth or decline, and interaction terms between firm characteristics and growth indicators

**Model Specifications for Fast-Growth Prediction**

| Specification | Included Variables |
|--------------|-------------------|
| Equation 1 | Firm age (linear and squared), foreign management, female ownership, municipality size, industry type |
| Equation 2 | All in Equation 1 + growth variables (e.g., sales, assets, expenses) |
| Equation 3 | All in Equation 2 + dummies for >2× growth (dummy_gt2x_*) |
| Equation 4 | All in Equation 3 + dummies for >–50% growth (dummy_drop50_*) |
| Equation 5 | All in Equation 4 + interaction terms (industry × growth, female × growth) |

In addition to standard logistic regression, a 'penalized' logistic regression with LASSO regularization is fit using the glmnet method with a grid of lambda values. The model is tuned using cross-validation to find the optimal penalty parameter. The coefficients for the best lambda are extracted and the RMSE results are added to the overall performance comparison.

Then, the random forest classifier is also trained. The model uses the ranger engine with a grid search over key hyperparameters (mtry, min.node.size) and the same 5-fold CV framework. The chosen hyperparameters are not too heavy on the model, but at the same time they produce better predictions relative to previous models:

* .mtry: controls the number of variables randomly selected as candidates at each split. In the grid, values of 5, 10, and 15 were explored. A smaller .mtry increases model diversity by forcing trees to consider different subsets of features, which often helps reduce overfitting.
* .splitrule: set to "gini", which determines how nodes are split by minimizing the Gini impurity. This criterion favors splits that result in purer child nodes.
* .min.node.size: defines the minimum number of observations that must exist in a terminal node. Values of 5 and 10 were tested, with smaller sizes allowing deeper trees that can capture more complex patterns, but at the risk of overfitting. 

For each model—particularly X4, X5, LASSO, and RF fold-wise AUC values are computed using ROC analysis. These are visualized with boxplots to show variation in AUC across folds. This gives a visual understanding of the average distribution of the results of each model. 
