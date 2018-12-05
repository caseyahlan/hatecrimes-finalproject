#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
alt.renderers.enable('notebook')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


# In[4]:


# Read in CSVs
hate_crimes_by_socioeconomic = pd.read_csv("./data/hate_crimes.csv")


# In[5]:


# Feature Selection
no_HI = hate_crimes_by_socioeconomic[hate_crimes_by_socioeconomic.state != 'Hawaii']
no_HI = no_HI.rename(columns={'hate_crimes_per_100k_splc': 'hate_crimes'})
no_HI = no_HI.rename(columns={'avg_hatecrimes_per_100k_fbi': 'avg_hate_crimes'})
array = no_HI.values
x = array[:,1:10]
y = array[:,10]
y = y.astype('float')
model = LinearRegression()
rfe = RFE(model)
fit = rfe.fit(x, y)

columns = "+".join(['share_unemployed_seasonal', 'share_population_with_high_school_degree', 'share_white_poverty', 'gini_index'])

multi_model = smf.ols('hate_crimes ~' + (columns), data = no_HI).fit()
multi_model.summary()


# In[7]:


no_HI['predictions_using_selected_features'] = multi_model.predict()
pred_actual_melt = no_HI[['state', 'hate_crimes', 'predictions_using_selected_features']]
pred_actual_melt = pred_actual_melt.melt('state', var_name='actual_vs_pred', value_name='value')

selected_features_actual_preds = alt.Chart(pred_actual_melt).mark_bar(opacity = 0.7).encode(
    alt.X('state', scale=alt.Scale(rangeStep=12), axis=alt.Axis(title='State')),
    alt.Y('value', axis=alt.Axis(title='Hate Crimes per 100k People', grid=False), stack = None),
    color=alt.Color('actual_vs_pred', scale=alt.Scale(range=["#EA98D2", "#659CCA"])),
).configure_view(
    stroke='transparent'
).configure_axis(
    domainWidth=0.8
).properties (
    title = 'Actual vs Predicted Number of Hate Crimes by State (Selected Features)'
).interactive();
selected_features_actual_preds


# In[8]:


# Modeling with all features
multi_model_all = smf.ols('hate_crimes ~ median_household_income + share_unemployed_seasonal + share_population_in_metro_areas + share_population_with_high_school_degree + share_non_citizen + share_white_poverty + gini_index + share_non_white + share_voters_voted_trump', data = no_HI).fit()
multi_model_all.summary()

no_HI['predictions_using_all_features'] = multi_model_all.predict()
pred_actual_melt_all = no_HI[['state', 'hate_crimes', 'predictions_using_all_features']]
pred_actual_melt_all = pred_actual_melt_all.melt('state', var_name='actual_vs_pred', value_name='value')

all_features_actual_preds = alt.Chart(pred_actual_melt_all).mark_bar(opacity = 0.7).encode(
    alt.X('state', scale=alt.Scale(rangeStep=12), axis=alt.Axis(title='State')),
    alt.Y('value', axis=alt.Axis(title='Hate Crimes per 100k People', grid=False), stack = None),
    color=alt.Color('actual_vs_pred', scale=alt.Scale(range=["#EA98D2", "#659CCA"]), legend = alt.Legend(title = 'Actual vs Predicted')),
).configure_view(
    stroke='transparent'
).configure_axis(
    domainWidth=0.8
).properties (
    title = 'Actual vs Predicted Number of Hate Crimes by State (All Features)'
).interactive();
all_features_actual_preds


# In[7]:


# Compares actual hate crime cases to predicted
pred_actual_hate_crime = plt.figure()
plt.scatter(no_HI.hate_crimes, no_HI.predictions_using_all_features)
plt.plot(no_HI.hate_crimes, no_HI.hate_crimes)
plt.title('Predicted vs. Actual Hate Crimes')
plt.xlabel('Actual Hate Crimes')
plt.ylabel('Predicted Hate Crimes');


# In[9]:


# Residual plot to assess accuracies
f, (plot1, plot2) = plt.subplots(1, 2, sharey=True)
f.set_figwidth(15)
plot1.scatter(no_HI.hate_crimes, no_HI.hate_crimes - no_HI.predictions_using_all_features)
plot1.axhline(0, color='red')
plot1.set_title('Residual Plot of Predictions Using All Features')
plot1.set_xlabel('Actual Hate Crimes')
plot1.set_ylabel('Actual - Predicted Hate Crimes')
plot2.scatter(no_HI.hate_crimes, no_HI.hate_crimes - no_HI.predictions_using_selected_features)
plot2.axhline(0, color='red')
plot2.set_title('Residual Plot of Predictions in Using Selected Features')
plot2.set_xlabel('Actual Hate Crimes')
plot2.set_ylabel('Actual - Predicted Hate Crimes');


# In[10]:


# Average hate crime cases from 2010 - 2015
multi_model_all = smf.ols('avg_hate_crimes ~ median_household_income + share_unemployed_seasonal + share_population_in_metro_areas + share_population_with_high_school_degree + share_non_citizen + share_white_poverty + gini_index + share_non_white + share_voters_voted_trump', data = no_HI).fit()
multi_model_all.summary()


# In[9]:


# Average hate crime cases from 2010 - 2015
no_HI['avg_predictions'] = multi_model_all.predict()
pred_actual_melt_all = no_HI[['state', 'avg_hate_crimes', 'avg_predictions']]
pred_actual_melt_all = pred_actual_melt_all.melt('state', var_name='actual_vs_pred', value_name='value')

actual_preds_by_state = alt.Chart(pred_actual_melt_all).mark_bar(opacity = 0.7).encode(
    alt.X('state', scale=alt.Scale(rangeStep=12), axis=alt.Axis(title='State')),
    alt.Y('value', axis=alt.Axis(title='Hate Crimes per 100k People', grid=False), stack = None),
    color=alt.Color('actual_vs_pred', scale=alt.Scale(range=["#EA98D2", "#659CCA"]), legend = alt.Legend(title = 'Actual vs Predicted')),
).configure_view(
    stroke='transparent'
).configure_axis(
    domainWidth=0.8
).properties (
    title = 'Actual vs Predicted Average Hate Crimes by State from 2010 - 2015'
).interactive();
actual_preds_by_state


# In[11]:


# Compares actual hate crime cases to predicted
fbi_actual_preds_hate_crimes = plt.figure()
plt.scatter(no_HI.avg_hate_crimes, no_HI.avg_predictions)
plt.plot(no_HI.avg_hate_crimes, no_HI.avg_hate_crimes)
plt.title('Predicted vs. Actual Average Hate Crimes from 2010 - 2015')
plt.xlabel('Actual Hate Crimes')
plt.ylabel('Predicted Hate Crimes');


# In[12]:


# Residual plot
fbi_residual_plot = plt.scatter(no_HI.avg_hate_crimes, no_HI.avg_hate_crimes - no_HI.avg_predictions)
plt.axhline(0, color='red')
plt.title('Residual Plot of Avg Predictions from 2010 - 2015')
plt.xlabel('Actual Average Hate Crimes')
plt.ylabel('Actual - Predicted Hate Crimes');


# In[ ]:




