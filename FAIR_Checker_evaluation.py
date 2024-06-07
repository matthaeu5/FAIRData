#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import re
import json
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
from itertools import chain
from matplotlib import pyplot as plt



# In[2]:


def get_all_metrics (payload):
    global count
    print(count)
    FAIR_checker_api = "https://fair-checker.france-bioinformatique.fr/api/check/metrics_all"
    try:
        response = requests.get(FAIR_checker_api, params=payload)
        print(payload)
        print(response.status_code)
        print("________________")
        count += 1
        return response.json(), response.status_code
    except:
        status = 504
        print(payload)
        print(status)
        print("________________")
        return None, status


# In[3]:


def get_metrics_from_result(assessment_result):
    return [item.get('metric') for item in assessment_result]

def get_scores_from_result(assessment_result):
    return [int(item.get('score')) for item in assessment_result]


# In[4]:


def get_principle_from_metric(metric):
    if metric.startswith('F'):
        return 'findable'
    elif metric.startswith('A'):
        return 'accessible'
    elif metric.startswith('I'):
        return 'interoperable'
    else:
        return 'reusable'


# In[5]:


def build_assessment_json(assessment):
    regex_result = re.findall("'metric': '.*?', 'score': '.*?'", assessment)
    assessment_list = []
    for result in regex_result:
        result = result.replace("\'", "\"")
        assessment_dict = json.loads("{"+result+"}")
        assessment_list.append(assessment_dict)
    return assessment_list


# In[6]:


def print_summary_stats(df):
    print("Mean: " + str(df["score"].mean()))
    print("Median: " + str(df["score"].median()))
    print("Max: " + str(df["score"].max()))
    print("Min: " + str(df["score"].min()))


# Metric descriptions as stated on the FAIR-Checker website: https://fair-checker.france-bioinformatique.fr/check
# - Findable
#     - F1A: Unique IDs - FAIRChecker check that the resource identifier is an URL that can be reach, meaning it is unique, it is even better if the URL refer to a DOI. 
#     - F1B: Persistent IDs - Weak : FAIR-Checker verifies that at least one namespace from identifiers.org is used in metadata. Strong : FAIR-Checker verifies that the “identifier” property from DCTerms or Schema.org vocabularies is present in metadata. 
#     - F2A: Structured metadata - FAIR-Checker verifies that at least one RDF triple can be found in metadata. 
#     - F2B: Shared vocabularies for metadata - Weak: FAIR-Checker verifies that at least one used ontology class or property are known in major ontology registries (OLS, BioPortal, LOV); Strong: FAIR-Checker verifies that all used ontology classes or properties are known in major ontology registries (OLS, BioPortal, LOV)
# - Accessible
#     - A1.1: Open resolution protocol - FAIR-Checker verifies that the resource is accessible via an open protocol, for instance the protocol needs to be HTTP. 
# - Interoperable
#     - I1: Machine readable format - FAIR-Checker verifies that at least one RDF triple can be found in metadata.
#     - I2: Use shared ontologies - Weak: FAIR-Checker verifies that at least one used ontology class or property are known in major ontology registries (OLS, BioPortal, LOV); Strong: FAIR-Checker verifies that all used ontology classes or properties are known in major ontology registries (OLS, BioPortal, LOV)
#     - I3: External links - FAIR-Checker verifies that at least 3 different URL authorities are used in the URIs of RDF metadata.
# - Reusable
#     - R1.1: Metadata includes license - FAIR-Checker verifies that at least one license property from Schema.org, DCTerms, or DOAP ontologies are found in metadata. 
#     - R1.2: Metadata includes provenance - FAIR-Checker verifies that at least one provenance property from PROV, DCTerms, or PAV ontologies are found in metadata. 
#     - R1.3: Community standards - Weak: FAIR-Checker verifies that at least one used ontology class or property are known in major ontology registries (OLS, BioPortal, LOV); Strong: FAIR-Checker verifies that all used ontology classes or properties are known in major ontology registries (OLS, BioPortal, LOV) 

# # Papers

# In[7]:


####################################

def extend_paper_df(df):
    ext_df = pd.DataFrame()
    
    ext_df["paper"] = list(chain.from_iterable([[x]*12 for x in df["paper"]]))
    ext_df["research_field"] = list(chain.from_iterable([x]*12 for x in df["research_field_label"]))
    ext_df["doi"] = list(chain.from_iterable([[x]*12 for x in df["doi"]]))
    ext_df["metric"] = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in df["FAIR_assessment"]]))
    ext_df["score"] = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in df["FAIR_assessment"]]))

    return ext_df


# ## Papers via DOI

# ### Paper assessment via DOI

# In[21]:


#############################################################################
input_file_path = r"C:\Users\MolnarM\Downloads\ogdmetadatendatagvat.csv"
output_file_path = r"C:\Users\MolnarM\Downloads\ogdmetadatendatagvat_cleaned.csv"

# Read the content of the file
with open(input_file_path, 'r') as file:
    content = file.read()

# Replace backslashes with forward slashes
modified_content = content.replace("\\", "/")

# Write the modified content back to a new file
with open(output_file_path, 'w') as file:
    file.write(modified_content)

print(f"Modified file saved as {output_file_path}")


# In[7]:


df = pd.read_csv(r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_cleaned.csv", encoding= 'unicode_escape', sep=';', on_bad_lines='skip', header=5, skiprows=[6,7])
df.head(10)


# In[9]:


######################################
get_ipython().run_line_magic('%script', 'false --no-raise-error')
orkg_doi_df = pd.read_csv(r"C:\Users\MolnarM\Downloads\orkg_papers_2023-03-29.csv", encoding='ISO-8859-1', index_col=0)
orkg_doi_df


# In[8]:


date = "2024-05-24"


# In[10]:


# FAIR-Checker assessment for all ORKG papers with a valid doi
# Executing this approximately takes 1,505 minutes
# If you don't have the time, use the latest .csv file :)
count = 0
assessment_results = df['metadata_identifier'].map(lambda x: get_all_metrics(payload={'url': 'https://www.data.gv.at/katalog/dataset/' + x}))
df['FAIR_assessment'] = [aresult[0] for aresult in assessment_results]
df['assessment_status_code'] = [aresult[1] for aresult in assessment_results]
df.to_csv(r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_assessed_"+date+".csv", encoding='utf-8')
df


# In[13]:


# Create a function to handle assessment and intermediate saving
def assess_and_save(df, save_interval=100, save_path=r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_assessed_intermediate.csv"):
    assessment_results = []
    
    for index, x in df['metadata_identifier'].iteritems():
        if pd.notna(x):
            result = get_all_metrics(payload={'url': 'https://www.data.gv.at/katalog/dataset/' + str(x)})
        else:
            result = (None, None)
        
        assessment_results.append(result)
        
        # Save intermediate results
        if (index + 1) % save_interval == 0:
            df['FAIR_assessment'] = [aresult[0] for aresult in assessment_results]
            df['assessment_status_code'] = [aresult[1] for aresult in assessment_results]
            df.to_csv(save_path, encoding='utf-8')
            print(f"Saved intermediate results at index {index + 1}")
            time.sleep(1)  # To avoid hitting API rate limits or overloading the server
    
    return assessment_results

# Load your DataFrame (assuming df is already loaded)
# df = pd.read_csv("path_to_your_file.csv")  # Uncomment and modify this line as needed

# Perform the assessment and save intermediate results
assessment_results = assess_and_save(df)

# Save the final results
date = "20230524"  # Replace with your desired date string
output_path = r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_assessed_" + date + ".csv"
df['FAIR_assessment'] = [aresult[0] for aresult in assessment_results]
df['assessment_status_code'] = [aresult[1] for aresult in assessment_results]
df.to_csv(output_path, encoding='utf-8')

# Display the DataFrame
print(df)


# In[15]:


import pandas as pd
import requests
import time

# Initialize the global count variable
count = 0

def get_all_metrics(payload):
    global count
    print(count)
    FAIR_checker_api = "https://fair-checker.france-bioinformatique.fr/api/check/metrics_all"
    try:
        response = requests.get(FAIR_checker_api, params=payload)
        print(payload)
        print(response.status_code)
        print("________________")
        count += 1
        return response.json(), response.status_code
    except:
        status = 504
        print(payload)
        print(status)
        print("________________")
        return None, status

def assess_and_save(df, save_interval=100, save_path=r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_assessed_intermediate.csv"):
    # Initialize columns with None
    df['FAIR_assessment'] = None
    df['assessment_status_code'] = None

    for index, x in df['metadata_identifier'].iteritems():
        if pd.notna(x):
            result = get_all_metrics(payload={'url': 'https://www.data.gv.at/katalog/dataset/' + str(x)})
        else:
            result = (None, None)
        
        # Update the DataFrame with the result
        df.at[index, 'FAIR_assessment'] = result[0]
        df.at[index, 'assessment_status_code'] = result[1]
        
        # Save intermediate results
        if (index + 1) % save_interval == 0:
            df.to_csv(save_path, encoding='utf-8', index=False)
            print(f"Saved intermediate results at index {index + 1}")
            time.sleep(1)  # To avoid hitting API rate limits or overloading the server
    
    return df

# Load your DataFrame (assuming df is already loaded)
# df = pd.read_csv("path_to_your_file.csv")  # Uncomment and modify this line as needed

# Perform the assessment and save intermediate results
df = assess_and_save(df)

# Save the final results
date = "20230527"  # Replace with your desired date string
output_path = r"C:\Users\MolnarM\Desktop\Notebooks\Uni\ogdmetadatendatagvat_assessed_" + date + ".csv"
df.to_csv(output_path, encoding='utf-8', index=False)


# In[18]:


df['FAIR_assessment'][0]


# In[19]:


df2 = df.copy()
df2.head()


# In[24]:


len(df2)


# In[46]:


df2 = df2.iloc[0:38556]


# In[47]:


df2.to_excel(r"C:\Users\MolnarM\Desktop\Notebooks\Uni\assessed_excel.xlsx")
df2.to_csv(r"C:\Users\MolnarM\Desktop\Notebooks\Uni\assessed_csv.csv")


# In[48]:


df2['assessment_status_code'].value_counts()


# In[23]:


df.to_excel(r"C:\Users\MolnarM\Desktop\Notebooks\Uni\assessed_excel.xlsx")


# In[21]:


df2['FAIR_assessment'] = df2['FAIR_assessment'].apply(eval)


# In[22]:


import pandas as pd
import ast



# Convert the string representation of list of dicts into actual list of dicts
def safe_eval(entry):
    if isinstance(entry, str):
        return ast.literal_eval(entry)
    return entry

df2['FAIR_assessment'] = df2['FAIR_assessment'].apply(safe_eval)

# Expand FAIR_assessment column
def expand_fair_assessment(row):
    metrics = {}
    for entry in row:
        metrics[entry['metric']] = entry['score']
    return pd.Series(metrics)

# Apply the function to each row
expanded_df = df2['FAIR_assessment'].apply(expand_fair_assessment)

# Concatenate the new columns with the original DataFrame
result_df = pd.concat([df2, expanded_df], axis=1).drop(columns=['FAIR_assessment'])

import ace_tools as tools; tools.display_dataframe_to_user(name="Expanded FAIR Assessment DataFrame", dataframe=result_df)

# Display the result
result_df


# In[20]:


# Expand FAIR_assessment column
def expand_fair_assessment(row):
    metrics = {}
    for entry in row:
        metrics[entry['metric']] = entry['score']
    return pd.Series(metrics)

# Apply the function to each row
expanded_df = df2['FAIR_assessment'].apply(expand_fair_assessment)

# Concatenate the new columns with the original DataFrame
result_df = pd.concat([df2, expanded_df], axis=1).drop(columns=['FAIR_assessment'])

import ace_tools as tools; tools.display_dataframe_to_user(name="Expanded FAIR Assessment DataFrame", dataframe=result_df)

# Display the result
result_df


# ### Trying to separate the metrics

# In[49]:


import pandas as pd
import ast



# Convert the string representation of list of dicts into actual list of dicts
def safe_eval(entry):
    if isinstance(entry, str):
        return ast.literal_eval(entry)
    return entry

df2['FAIR_assessment'] = df2['FAIR_assessment'].apply(safe_eval)


# In[53]:


df2['FAIR_assessment'][0]


# In[51]:


# Expand FAIR_assessment column into separate metric columns
def expand_fair_assessment(row):
    metrics = {}
    for entry in row:
        metrics[entry['metric']] = entry['score']
    return pd.Series(metrics)


# In[69]:


for i in df2['FAIR_assessment'].tolist()[110]:
    print(i['metric'], i['score'])
    print(i)
    


# In[52]:


expanded_df = df['FAIR_assessment'].apply(expand_fair_assessment)


# In[70]:


import pandas as pd

# Sample DataFrame
data = {
    'metrics': [
        [{'metric': 'A', 'score': 10}, {'metric': 'B', 'score': 20}, {'metric': 'C', 'score': 30}],
        [{'metric': 'A', 'score': 15}, {'metric': 'B', 'score': 25}, {'metric': 'C', 'score': 35}],
        [{'metric': 'A', 'score': 20}, {'metric': 'B', 'score': 30}, {'metric': 'C', 'score': 40}]
    ]
}

dfx = pd.DataFrame(data)


# In[73]:


get_ipython().system('pip install ace')


# In[82]:


df3 = df2[df2['assessment_status_code']==200]


# In[118]:


df3


# In[83]:


def convert_metrics_to_dict(metric_list):
    return {d['metric']: d['score'] for d in metric_list}

# Apply the function to the 'metrics' column
metrics_dicts = df3['FAIR_assessment'].apply(convert_metrics_to_dict)

# Create a new DataFrame from the list of dictionaries
metrics_df = pd.DataFrame(metrics_dicts.tolist())

# Display the result
#import ace_tools as tools; tools.display_dataframe_to_user(name="Metrics DataFrame", dataframe=metrics_df)

metrics_df


# In[119]:


metrics_df.value_counts()


# In[87]:


metrics_df['F1B'].unique()


# In[100]:


metrics_df.apply(pd.to_numeric)


# In[117]:


metrics_df.iloc[[13859]]


# In[108]:


df_combined = pd.concat([df3, metrics_df], axis=1, ignore_index=True)


# In[109]:


df_combined.tail(15)


# In[ ]:


# Concatenate the new columns with the original DataFrame
result_df = pd.concat([df, expanded_df], axis=1)

# Ensure only rows with status_code of 200 have values; others should be NaN
result_df.loc[result_df['status_code'] != 200, expanded_df.columns] = None
# using the .csv from the doi paper assessments

# Drop the original FAIR_assessment column
result_df.drop(columns=['FAIR_assessment'], inplace=True)

import ace_tools as tools; tools.display_dataframe_to_user(name="Expanded FAIR Assessment DataFrame", dataframe=result_df)

# Display the result
result_df


# ### Paper evaluation via DOI + visualisations

# In[10]:


# using the .csv from the doi paper assessments
# only keeping successful assessments
orkg_doi_df = pd.read_csv(r"C:\Users\MolnarM\Downloads\orkg_stdurl_df_FAIRCheckerv2_assessment2023-03-29.csv", encoding='utf-8', index_col=0)
print(orkg_doi_df["assessment_status_code"].value_counts())
orkg_doi_df = orkg_doi_df[orkg_doi_df["assessment_status_code"] == 200]
orkg_doi_df = orkg_doi_df.drop(labels=["assessment_status_code"], axis=1).reset_index(drop=True)
orkg_doi_df["FAIR_assessment"] = orkg_doi_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
orkg_doi_df


# In[13]:


assessment_df


# In[11]:


assessment_df = extend_paper_df(orkg_doi_df)
assessment_df


# In[12]:


aggregate_funcs = {'research_field': 'first', 'doi': 'first', 'score': 'sum'}
total_score_df = assessment_df.groupby('paper', as_index=False).agg(aggregate_funcs)
total_score_df


# In[13]:


print_summary_stats(total_score_df)


# In[14]:


eval1_df = total_score_df.groupby("research_field").agg({"score": "mean"})
eval1_df["counts"] = total_score_df["research_field"].value_counts()
eval1_df = eval1_df.sort_values("counts").reset_index()

plot = plt.figure()
ax = plot.add_axes([0,0,1,1])
fields = eval1_df["research_field"]
avg_scores = eval1_df["score"]
counts_color = [{p<5: "red", 5<=p<30: "orange", 30<=p<=100: "lightgreen", p>100: "green"}[True] for p in eval1_df["counts"]]
ax.barh(fields, avg_scores, color=counts_color)
leg_red = mlines.Line2D([], [], color="red", marker="s", ls="", label="<5 papers")
leg_orange = mlines.Line2D([], [], color="orange", marker="s", ls="", label=">5 and <29 papers")
leg_lightgreen = mlines.Line2D([], [], color="lightgreen", marker="s", ls="", label=">30 and <99 papers")
leg_green = mlines.Line2D([], [], color="green", marker="s", ls="", label=">100 papers")
plt.legend(handles=[leg_green, leg_lightgreen, leg_orange, leg_red])
plt.title("Average FAIRness score for research fields (FAIR-Checker) - DOI assessement")
plt.show()


# In[15]:


# max scores: F = 8, A = 4, I = 6, R = 6
principle_assessment_df = assessment_df
principle_assessment_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_df = principle_assessment_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_df['avg_metric_score'] = principle_assessment_df['score']/len(orkg_doi_df)

principle_assessment_df['principle'] = pd.Categorical(principle_assessment_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_df = principle_assessment_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_df['principle'] )
plt.title("Average score for FAIR principles - DOI assessment")
plt.legend()
plt.show()


# In[16]:


metric_assessment_doi_df = assessment_df
aggregate_funcs = {"score": "sum"}
metric_assessment_doi_df = metric_assessment_doi_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_doi_df["score"] = metric_assessment_doi_df["score"] / 2

metric_assessment_doi_df['metric'] = pd.Categorical(metric_assessment_doi_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_doi_df = metric_assessment_doi_df.sort_values('metric')

metric_assessment_doi_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by papers (DOI assessment)")
plt.ylabel("#papers that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()


# ## Papers via ORKG resource URL

# ### Paper assessment via ORKG resource URL

# In[17]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', 'orkg_url_df = pd.read_csv("raw_data/paper_query_result_2023-03-29.csv", encoding="ISO-8859-1")\norkg_url_df = orkg_url_df[["paper", "paper_title", "url", "research_field_label"]]\norkg_url_df = orkg_url_df.groupby("paper").agg({"paper_title": "first", "url": "first", "research_field_label": "first"}).reset_index()\norkg_url_df = orkg_url_df.drop(labels=["url"], axis=1)\norkg_url_df\n')


# In[18]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '# FAIR-Checker assessment for all ORKG papers with an ORKG resource URL\n# Executing this approximately takes 1,716 minutes\n# If you don\'t have the time, use the latest .csv file :)\ncount = 0\nassessment_results = orkg_url_df[\'paper\'].map(lambda x: get_all_metrics(payload={\'url\': x}))\norkg_url_df[\'FAIR_assessment\'] = [aresult[0] for aresult in assessment_results]\norkg_url_df[\'assessment_status_code\'] = [aresult[1] for aresult in assessment_results]\norkg_url_df.to_csv("./assessed_data/orkg_url_df_FAIRCheckerv2_assessment"+date+".csv")\norkg_url_df\n')


# ### Paper evaluation via ORKG resource URL + visualizations

# In[19]:


orkg_url_df = pd.read_csv("assessed_data/orkg_url_df_FAIRCheckerv2_assessment2023-03-29.csv", encoding='ISO-8859-1', index_col=0)
print(orkg_url_df["assessment_status_code"].value_counts())
orkg_url_df = orkg_url_df[orkg_url_df["assessment_status_code"] == 200].reset_index(drop=True)
orkg_url_df["FAIR_assessment"] = orkg_url_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
orkg_url_df


# In[20]:


assessment_orkgurl_df = pd.DataFrame()

papers = list(chain.from_iterable([[x]*12 for x in orkg_url_df['paper']]))
assessment_orkgurl_df['paper'] = papers

research_fields = list(chain.from_iterable([[x]*12 for x in orkg_url_df['research_field_label']]))
assessment_orkgurl_df['research_field'] = research_fields

metrics = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in orkg_url_df['FAIR_assessment']]))
assessment_orkgurl_df['metric'] = metrics

scores = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in orkg_url_df['FAIR_assessment']]))
assessment_orkgurl_df['score'] = scores

assessment_orkgurl_df


# In[21]:


aggregate_funcs = {'research_field': 'first', 'score': 'sum'}
total_score_orkgurl_df = assessment_orkgurl_df.groupby('paper', as_index=False).agg(aggregate_funcs)
total_score_orkgurl_df


# In[22]:


print(total_score_orkgurl_df["score"].mean())
print(total_score_orkgurl_df["score"].median())
# possible max = 16
print(total_score_orkgurl_df["score"].max())
# possible min = 2
print(total_score_orkgurl_df["score"].min())


# In[23]:


eval2_df = total_score_orkgurl_df.groupby("research_field").agg({"score": "sum"})
eval2_df["counts"] = total_score_orkgurl_df["research_field"].value_counts()
eval2_df = eval2_df.sort_values("counts")
eval2_df["avg_score"] = eval2_df["score"] / eval2_df["counts"]
eval2_df = eval2_df.drop(labels=["score"], axis=1)
eval2_df = eval2_df.reset_index()

plot = plt.figure()
ax = plot.add_axes([0,0,1,1])
fields = eval2_df['research_field']
avg_scores = eval2_df['avg_score']
counts_color = [{p<5: 'red', 5<=p<30: 'orange', 30<=p<=100: 'lightgreen', p>100: 'green'}[True] for p in eval2_df["counts"]]
ax.barh(fields, avg_scores, color=counts_color)
leg_red = mlines.Line2D([], [], color="red", marker="s", ls='', label="<5 papers")
leg_orange = mlines.Line2D([], [], color="orange", marker="s", ls='', label=">5 and <29 papers")
leg_lightgreen = mlines.Line2D([], [], color="lightgreen", marker="s", ls='', label=">30 and <99 papers")
leg_green = mlines.Line2D([], [], color="green", marker="s", ls='', label=">100 papers")
plt.legend(handles=[leg_green, leg_lightgreen, leg_orange, leg_red])
plt.title("Average FAIRness score for research fields (FAIR-Checker) - ORKG resource URL assessment")
plt.show()


# In[24]:


# max scores: F = 8, A = 2, I = 6, R = 6
principle_assessment_orkgurl_df = assessment_orkgurl_df
principle_assessment_orkgurl_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_orkgurl_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_orkgurl_df = principle_assessment_orkgurl_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_orkgurl_df['avg_metric_score'] = principle_assessment_orkgurl_df['score']/len(orkg_url_df)

principle_assessment_orkgurl_df['principle'] = pd.Categorical(principle_assessment_orkgurl_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_orkgurl_df = principle_assessment_orkgurl_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_orkgurl_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_orkgurl_df['principle'] )
plt.title("Average score for FAIR principles - ORKG resource URL assessment")
plt.legend()
plt.show()


# In[25]:


metric_assessment_orkgurl_df = assessment_orkgurl_df
aggregate_funcs = {"score": "sum"}
metric_assessment_orkgurl_df = metric_assessment_orkgurl_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_orkgurl_df["score"] = metric_assessment_orkgurl_df["score"] / 2

metric_assessment_orkgurl_df['metric'] = pd.Categorical(metric_assessment_orkgurl_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_orkgurl_df = metric_assessment_orkgurl_df.sort_values('metric')

metric_assessment_orkgurl_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by papers (ORKG URL assessment)")
plt.ylabel("#papers that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()


# ## Papers via standard URL

# ### Paper assessment via standard URL

# In[26]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', 'orkg_stdurl_df = pd.read_csv("raw_data/paper_query_result_2023-03-29.csv", encoding="ISO-8859-1", index_col=0).reset_index()\norkg_stdurl_df = orkg_stdurl_df[["paper", "paper_title", "url", "research_field_label"]]\norkg_stdurl_df = orkg_stdurl_df.groupby("paper").agg({"paper_title": "first", "url": "first", "research_field_label": "first"})\norkg_stdurl_df = orkg_stdurl_df[orkg_stdurl_df["url"].isna() == False].reset_index()\norkg_stdurl_df\n')


# In[27]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '# FAIR-Checker assessment for all ORKG papers with a standard URL\n# Executing this approximately takes 2,471 minutes\n# If you don\'t have the time, use the latest .csv file :)\ncount = 0\nassessment_results = orkg_stdurl_df[\'url\'].map(lambda x: get_all_metrics(payload={\'url\': x}))\norkg_stdurl_df[\'FAIR_assessment\'] = [aresult[0] for aresult in assessment_results]\norkg_stdurl_df[\'assessment_status_code\'] = [aresult[1] for aresult in assessment_results]\norkg_stdurl_df.to_csv("./assessed_data/orkg_stdurl_df_FAIRCheckerv2_assessment"+date+".csv")\norkg_stdurl_df\n')


# ### Paper evaluation via standard URL + visualizations

# In[28]:


orkg_stdurl_df = pd.read_csv("assessed_data/orkg_stdurl_df_FAIRCheckerv2_assessment2023-03-29.csv", encoding="ISO-8859-1", index_col=0)
print(orkg_stdurl_df["assessment_status_code"].value_counts())
orkg_stdurl_df = orkg_stdurl_df[orkg_stdurl_df["assessment_status_code"] == 200].reset_index(drop=True)
orkg_stdurl_df["FAIR_assessment"] = orkg_stdurl_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
orkg_stdurl_df


# In[29]:


assessment_stdurl_df = pd.DataFrame()

papers = list(chain.from_iterable([[x]*12 for x in orkg_stdurl_df['paper']]))
assessment_stdurl_df['paper'] = papers

research_fields = list(chain.from_iterable([[x]*12 for x in orkg_stdurl_df['research_field_label']]))
assessment_stdurl_df['research_field'] = research_fields

urls = list(chain.from_iterable([[x]*12 for x in orkg_stdurl_df['url']]))
assessment_stdurl_df['url'] = urls

metrics = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in orkg_stdurl_df['FAIR_assessment']]))
assessment_stdurl_df['metric'] = metrics

scores = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in orkg_stdurl_df['FAIR_assessment']]))
assessment_stdurl_df['score'] = scores

assessment_stdurl_df


# In[30]:


aggregate_funcs = {'research_field': 'first', 'url': 'first', 'score': 'sum'}
total_score_stdurl_df = assessment_stdurl_df.groupby('paper', as_index=False).agg(aggregate_funcs)
total_score_stdurl_df


# In[31]:


print(total_score_stdurl_df["score"].mean())
print(total_score_stdurl_df["score"].median())
# possible max = 16
print(total_score_stdurl_df["score"].max())
# possible min = 2
print(total_score_stdurl_df["score"].min())


# In[32]:


eval3_df = total_score_stdurl_df.groupby("research_field").agg({"score": "sum"})
eval3_df["counts"] = total_score_stdurl_df["research_field"].value_counts()
eval3_df = eval3_df.sort_values("counts")
eval3_df["avg_score"] = eval3_df["score"] / eval3_df["counts"]
eval3_df = eval3_df.drop(labels=["score"], axis=1)
eval3_df = eval3_df.reset_index()

plot = plt.figure()
ax = plot.add_axes([0,0,1,1])
fields = eval3_df['research_field']
avg_scores = eval3_df['avg_score']
counts_color = [{p<5: 'red', 5<=p<30: 'orange', 30<=p<=100: 'lightgreen', p>100: 'green'}[True] for p in eval3_df["counts"]]
ax.barh(fields, avg_scores, color=counts_color)
leg_red = mlines.Line2D([], [], color="red", marker="s", ls='', label="<5 papers")
leg_orange = mlines.Line2D([], [], color="orange", marker="s", ls='', label=">5 and <29 papers")
leg_lightgreen = mlines.Line2D([], [], color="lightgreen", marker="s", ls='', label=">30 and <99 papers")
leg_green = mlines.Line2D([], [], color="green", marker="s", ls='', label=">100 papers")
plt.legend(handles=[leg_green, leg_lightgreen, leg_orange, leg_red])
plt.title("Average FAIRness score for research fields (FAIR-Checker) - standard URL assessment")
plt.show()


# In[33]:


# max scores: F = 8, A = 4, I = 6, R = 6
principle_assessment_stdurl_df = assessment_stdurl_df
principle_assessment_stdurl_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_stdurl_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_stdurl_df = principle_assessment_stdurl_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_stdurl_df['avg_metric_score'] = principle_assessment_stdurl_df['score']/len(orkg_stdurl_df)

principle_assessment_stdurl_df['principle'] = pd.Categorical(principle_assessment_stdurl_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_stdurl_df = principle_assessment_stdurl_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_stdurl_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_stdurl_df['principle'] )
plt.title("Average score for FAIR principles - standard URL assessment")
plt.legend()
plt.show()


# In[34]:


metric_assessment_stdurl_df = assessment_stdurl_df
aggregate_funcs = {"score": "sum"}
metric_assessment_stdurl_df = metric_assessment_stdurl_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_stdurl_df["score"] = metric_assessment_stdurl_df["score"] / 2

metric_assessment_stdurl_df['metric'] = pd.Categorical(metric_assessment_stdurl_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_stdurl_df = metric_assessment_stdurl_df.sort_values('metric')

metric_assessment_stdurl_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by papers (Standard URL assessment)")
plt.ylabel("#papers that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()


# ## Comparison of Papers: DOI vs. ORKG resource URL vs. standard URL

# In[35]:


doi_df = total_score_df
orkg_resource_url_df = total_score_orkgurl_df
std_url_df = total_score_stdurl_df

assessement_data = [doi_df["score"], orkg_resource_url_df["score"], std_url_df["score"]]
plt.boxplot(assessement_data)
plt.title("FAIR-Checker assessments for Papers in the ORKG (different populations)")
locs, labels = plt.xticks()
plt.xticks(locs, ["DOI", "ORKG URL", "std URL"])
plt.ylabel("average FAIRness score")
plt.show()


# In[36]:


doi_orkg_df = pd.merge(doi_df, orkg_resource_url_df, on=["paper", "research_field"], suffixes=("_doi", "_orkg"))
doi_orkg_stdurl_df = pd.merge(doi_orkg_df, std_url_df, on=["paper", "research_field"])
doi_orkg_stdurl_df = doi_orkg_stdurl_df.rename(columns={"score": "score_url"})
doi_orkg_stdurl_df = doi_orkg_stdurl_df[["paper", "research_field", "doi", "url", "score_doi", "score_orkg", "score_url"]]
doi_orkg_stdurl_df


# In[37]:


assessement_data_merged = [doi_orkg_stdurl_df["score_doi"], doi_orkg_stdurl_df["score_orkg"], doi_orkg_stdurl_df["score_url"]]
plt.boxplot(assessement_data_merged)
plt.title("FAIR-Checker assessments for Papers in the ORKG (equal populations)")
locs, labels = plt.xticks()
plt.xticks(locs, ["DOI", "ORKG URL", "std URL"])
plt.ylabel("average FAIRness score")
plt.show()


# # Comparisons

# ## Comparisons evaluation via DOI

# ### Comparison assessment via DOI

# In[40]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', 'comparison_df = pd.read_csv("raw_data/comparison_query_result_2023-03-29.csv")\ncomparison_doi_df = comparison_df[comparison_df["doi"].isna() == False].reset_index(drop=True)\ncomparison_doi_df\n')


# In[41]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '# FAIR-Checker assessment for all ORKG comparisons with a valid doi\n# Executing this approximately takes 25 minutes\n# If you don\'t have the time, use the latest .csv file :)\ncount = 0\nassessment_results = comparison_doi_df[\'doi\'].map(lambda x: get_all_metrics(payload={\'url\': \'https://doi.org/\' + x}))\ncomparison_doi_df[\'FAIR_assessment\'] = [aresult[0] for aresult in assessment_results]\ncomparison_doi_df[\'assessment_status_code\'] = [aresult[1] for aresult in assessment_results]\ncomparison_doi_df.to_csv("./assessed_data/comparison_doi_df_FAIRCheckerv2_assessment_"+date+".csv")\ncomparison_doi_df\n')


# ### Comparison evaluation via DOI + visualizations

# In[46]:


comparison_doi_df = pd.read_csv("assessed_data/comparison_doi_df_FAIRCheckerv2_assessment_2023-03-29.csv", index_col=0)
print(comparison_doi_df["assessment_status_code"].value_counts())
comparison_doi_df = comparison_doi_df[comparison_doi_df["assessment_status_code"] == 200]
comparison_doi_df = comparison_doi_df.drop(labels=["assessment_status_code"], axis=1)
comparison_doi_df["FAIR_assessment"] = comparison_doi_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
comparison_doi_df


# In[47]:


assessment_compdoi_df = pd.DataFrame()

comparisons = list(chain.from_iterable([[x]*12 for x in comparison_doi_df['comparisons']]))
assessment_compdoi_df['comparison'] = comparisons

dois = list(chain.from_iterable([[x]*12 for x in comparison_doi_df['doi']]))
assessment_compdoi_df['doi'] = dois

metrics = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in comparison_doi_df['FAIR_assessment']]))
assessment_compdoi_df['metric'] = metrics

scores = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in comparison_doi_df['FAIR_assessment']]))
assessment_compdoi_df['score'] = scores

assessment_compdoi_df


# In[48]:


aggregate_funcs = {'doi': 'first', 'score': 'sum'}
total_score_compdoi_df = assessment_compdoi_df.groupby('comparison', as_index=False).agg(aggregate_funcs)
total_score_compdoi_df


# In[49]:


print(total_score_compdoi_df["score"].mean())
print(total_score_compdoi_df["score"].median())
# possible max = 16
print(total_score_compdoi_df["score"].max())
# possible min = 16
print(total_score_compdoi_df["score"].min())


# In[50]:


# max scores: F = 8, A = 4, I = 6, R = 6
principle_assessment_compdoi_df = assessment_compdoi_df
principle_assessment_compdoi_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_compdoi_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_compdoi_df = principle_assessment_compdoi_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_compdoi_df['avg_metric_score'] = principle_assessment_compdoi_df['score']/len(comparison_doi_df)

principle_assessment_compdoi_df['principle'] = pd.Categorical(principle_assessment_compdoi_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_compdoi_df = principle_assessment_compdoi_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_compdoi_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_compdoi_df['principle'] )
plt.title("Average score for FAIR principles - DOI assessment")
plt.legend()
plt.show()


# In[51]:


metric_assessment_compdoi_df = assessment_compdoi_df
aggregate_funcs = {"score": "sum"}
metric_assessment_compdoi_df = metric_assessment_compdoi_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_compdoi_df["score"] = metric_assessment_compdoi_df["score"] / 2

metric_assessment_compdoi_df['metric'] = pd.Categorical(metric_assessment_compdoi_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_compdoi_df = metric_assessment_compdoi_df.sort_values('metric')

metric_assessment_compdoi_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by comparisons (DOI assessment)")
plt.ylabel("#comparisons that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()


# ## Comparisons evaluation via ORKG resource URL

# ### Comparison assessment via ORKG resource URL

# In[56]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', 'comparison_orkg_df = pd.read_csv("raw_data/comparison_query_result_2023-03-29.csv")\ncomparison_orkg_df\n')


# In[57]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '# FAIR-Checker assessment for all ORKG comparisons with an ORKG resource URL\n# Executing this approximately takes 1,505 minutes\n# If you don\'t have the time, use the latest .csv file :)\ncount = 0\nassessment_results = comparison_orkg_df[\'comparisons\'].map(lambda x: get_all_metrics(payload={\'url\': x}))\ncomparison_orkg_df[\'FAIR_assessment\'] = [aresult[0] for aresult in assessment_results]\ncomparison_orkg_df[\'assessment_status_code\'] = [aresult[1] for aresult in assessment_results]\ncomparison_orkg_df.to_csv("./assessed_data/comparison_orkg_df_FAIRCheckerv2_assessment"+date+".csv")\ncomparison_orkg_df\n')


# ### Comparison evaluation via ORKG resource URL

# In[55]:


comparison_orkg_df = pd.read_csv("assessed_data/comparison_orkg_df_FAIRCheckerv2_assessment2023-03-29.csv", index_col=0)
print(comparison_orkg_df["assessment_status_code"].value_counts())
comparison_orkg_df = comparison_orkg_df[comparison_orkg_df["assessment_status_code"] == 200].reset_index(drop=True)
comparison_orkg_df["FAIR_assessment"] = comparison_orkg_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
comparison_orkg_df


# In[58]:


assessment_comporkg_df = pd.DataFrame()

comparisons = list(chain.from_iterable([[x]*12 for x in comparison_orkg_df['comparisons']]))
assessment_comporkg_df['comparison'] = comparisons

metrics = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in comparison_orkg_df['FAIR_assessment']]))
assessment_comporkg_df['metric'] = metrics

scores = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in comparison_orkg_df['FAIR_assessment']]))
assessment_comporkg_df['score'] = scores

assessment_comporkg_df


# In[59]:


aggregate_funcs = {'score': 'sum'}
total_score_comporkg_df = assessment_comporkg_df.groupby('comparison', as_index=False).agg(aggregate_funcs)
total_score_comporkg_df


# In[60]:


print(total_score_comporkg_df["score"].mean())
print(total_score_comporkg_df["score"].median())
# possible max = 16
print(total_score_comporkg_df["score"].max())
# possible min = 16
print(total_score_comporkg_df["score"].min())


# In[61]:


# max scores: F = 8, A = 4, I = 6, R = 6
principle_assessment_comporkg_df = assessment_comporkg_df
principle_assessment_comporkg_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_comporkg_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_comporkg_df = principle_assessment_comporkg_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_comporkg_df['avg_metric_score'] = principle_assessment_comporkg_df['score']/len(comparison_orkg_df)

principle_assessment_comporkg_df['principle'] = pd.Categorical(principle_assessment_comporkg_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_comporkg_df = principle_assessment_comporkg_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_comporkg_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_comporkg_df['principle'] )
plt.title("Average score for FAIR principles - DOI assessment")
plt.legend()
plt.show()


# In[62]:


metric_assessment_comporkg_df = assessment_comporkg_df
aggregate_funcs = {"score": "sum"}
metric_assessment_comporkg_df = metric_assessment_comporkg_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_comporkg_df["score"] = metric_assessment_comporkg_df["score"] / 2

metric_assessment_comporkg_df['metric'] = pd.Categorical(metric_assessment_comporkg_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_comporkg_df = metric_assessment_comporkg_df.sort_values('metric')

metric_assessment_comporkg_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by comparisons (ORKG URL assessment)")
plt.ylabel("#comparisons that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()


# ## Comparison of Comparisons: DOI vs. ORKG resource URL

# In[63]:


compdoi_df = total_score_compdoi_df
comporkg_df = total_score_comporkg_df

assessement_data = [compdoi_df["score"], comporkg_df["score"]]
plt.boxplot(assessement_data)
plt.title("FAIR-Checker assessments for Papers in the ORKG (different populations)")
locs, labels = plt.xticks()
plt.xticks(locs, ["DOI", "ORKG URL"])
plt.ylabel("average FAIRness score")
plt.show()


# # Resources

# ## Resource assessment

# In[66]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', 'resource_df = pd.read_csv("raw_data/resource_query_result_2023-04-05.csv")\nresource_rsrc_df = resource_df[resource_df["type"].str.endswith("Resource")]\nresource_others_df = resource_df[np.logical_not(resource_df["type"].str.endswith("Resource"))]\nresource_df = pd.merge(resource_rsrc_df, resource_others_df, on="resources", how="outer", suffixes=("_rsrc", "_other"))\nresource_df = resource_df[np.logical_not(resource_df["type_other"].str.endswith("Paper") | resource_df["type_other"].str.endswith("Comparison"))].reset_index(drop=True)\nresource_df = resource_df.sample(n=3000, random_state=13)\nresource_df\n')


# In[67]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '# FAIR-Checker assessment for random sample ORKG resources that are not Papers or Comparisons with their ORKG resource URL\n# Executing this will take a long time\n# If you don\'t have the time, use the latest .csv file :)\ncount = 0\nassessment_results = resource_df[\'resources\'].map(lambda x: get_all_metrics(payload={\'url\': x}))\nresource_df[\'FAIR_assessment\'] = [aresult[0] for aresult in assessment_results]\nresource_df[\'assessment_status_code\'] = [aresult[1] for aresult in assessment_results]\nresource_df.to_csv("./assessed_data/resource_df_FAIRCheckerv2_assessment_2023-04-05.csv")\nresource_df\n')


# ## Resource evaluation + visualization

# In[69]:


resource_sample_df = pd.read_csv("assessed_data/resource_df_FAIRCheckerv2_assessment_2023-04-05.csv", index_col=0)
print(resource_sample_df["assessment_status_code"].value_counts())
resource_sample_df = resource_sample_df[resource_sample_df["assessment_status_code"] == 200].reset_index(drop=True)
resource_sample_df["FAIR_assessment"] = resource_sample_df["FAIR_assessment"].map(lambda x: build_assessment_json(x))
resource_sample_df


# In[70]:


assessment_resource_df = pd.DataFrame()

resources = list(chain.from_iterable([[x]*12 for x in resource_sample_df['resources']]))
assessment_resource_df['resource'] = resources

types_rsrcs = list(chain.from_iterable([[x]*12 for x in resource_sample_df['type_rsrc']]))
assessment_resource_df["type_rsrc"] = types_rsrcs

types_others = list(chain.from_iterable([[x]*12 for x in resource_sample_df['type_other']]))
assessment_resource_df["type_other"] = types_others

metrics = list(chain.from_iterable([get_metrics_from_result(assessment_result=x) for x in resource_sample_df['FAIR_assessment']]))
assessment_resource_df['metric'] = metrics

scores = list(chain.from_iterable([get_scores_from_result(assessment_result=x) for x in resource_sample_df['FAIR_assessment']]))
assessment_resource_df['score'] = scores

assessment_resource_df


# In[71]:


aggregate_funcs = {'score': 'sum', 'type_rsrc': 'first', 'type_other': 'first'}
total_score_resource_df = assessment_resource_df.groupby('resource', as_index=False).agg(aggregate_funcs)
total_score_resource_df


# In[72]:


print(total_score_comporkg_df["score"].mean())
print(total_score_comporkg_df["score"].median())
# possible max = 16
print(total_score_comporkg_df["score"].max())
# possible min = 16
print(total_score_comporkg_df["score"].min())


# In[73]:


# max scores: F = 8, A = 4, I = 6, R = 6
principle_assessment_resource_df = assessment_resource_df
principle_assessment_resource_df['principle'] = [get_principle_from_metric(x) for x in principle_assessment_resource_df['metric']]

aggregate_funcs = {'score': 'sum', 'principle': 'first'}
principle_assessment_resource_df = principle_assessment_resource_df.groupby(['principle'], as_index=False).agg(aggregate_funcs)
principle_assessment_resource_df['avg_metric_score'] = principle_assessment_resource_df['score']/len(resource_sample_df)

principle_assessment_resource_df['principle'] = pd.Categorical(principle_assessment_resource_df['principle'], ['findable', 'accessible', 'interoperable', 'reusable'])
principle_assessment_resource_df = principle_assessment_resource_df.sort_values('principle')

#plotting
width = 0.5
perfect_scores = [8, 4, 6, 6]
actual_scores = principle_assessment_resource_df['avg_metric_score']
indices = np.arange(len(perfect_scores))

plt.bar(indices, perfect_scores, width=width, color='b', alpha=0.5, label='Perfect Score')
plt.bar([i for i in indices], actual_scores, width=0.5*width, color='orange', label='Average Score')
plt.xticks(indices, principle_assessment_resource_df['principle'] )
plt.title("Average score for FAIR principles - DOI assessment")
plt.legend()
plt.show()


# In[74]:


metric_assessment_resource_df = assessment_resource_df
aggregate_funcs = {"score": "sum"}
metric_assessment_resource_df = metric_assessment_resource_df.groupby("metric").agg(aggregate_funcs).reset_index()
metric_assessment_resource_df["score"] = metric_assessment_resource_df["score"] / 2

metric_assessment_resource_df['metric'] = pd.Categorical(metric_assessment_resource_df['metric'], ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
metric_assessment_resource_df = metric_assessment_resource_df.sort_values('metric')

metric_assessment_resource_df["score"].plot(kind="bar")
locs, labels = plt.xticks()
plt.xticks(locs, ["F1A", "F1B", "F2A", "F2B", "A1.1", "A1.2", "I1", "I2", "I3", "R1.1", "R1.2", "R1.3"])
plt.title("FAIR-Checker metrics passed by comparisons (ORKG URL assessment)")
plt.ylabel("#comparisons that passed the metric")
plt.xlabel("FAIR-Checker metrics")
plt.show()

