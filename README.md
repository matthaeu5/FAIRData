# Github Repository
https://github.com/matthaeu5/FAIRData

# FAIR Data Evaluation of DACH's and Ireland's Open Data Strategy

This project evaluates the adherence to FAIR Data principles in DACH region and Ireland regarding their Open Data Strategy. The FAIR Data principles aim to enhance data value by making it Findable, Accessible, Interoperable, and Reusable. Our research focuses on assessing the FAIRness of data available on the respective open data portals and analyzing the results to provide insights and a qualitative comparison of the countries FAIR Data adherence.

# How to use this repository
<strong> REMINDER: you might have to change file / folder dependencies to proper run the code on your environment. </strong>
1. "DE, IR, CH, Open Data Portal metadata.ipynb" is for downloading the relevant metadata from Open Data Portals (of Germany, Ireland, and Switzerland). Outputs are excel files for each country. 
2. "API Assessment Process.ipynb" takes the downloaded metadata files and runs the assessment for those three countries.
3. "FAIR_Checker_Cleaned_vFinal.ipynb" does the assessment for the files downloaded from the Austrian data portal (not included in the metadata notebook as these files were downloaded directly from the website with no requests) and performs comparative analysis. Intermediate point and corresponding files are included where one can skip the lengthy assessment-process and run only the analysis.


## Project Overview

### Background
Introduced in 2016, the FAIR Data principles have become a cornerstone in improving data management and sharing practices. These principles are essential for increasing transparency, accountability, and efficiency in various sectors including research, healthcare, and urban planning.

### Research Question
How FAIR is the Open Data Strategy of DACH and Ireland?

### Sub Question
Which country adheres the most to FAIR Data principles?

### Methodology
1. **Data Collection**: Gather data from the open data portals.
2. **FAIRness Evaluation**: Perform a FAIRness evaluation using an existing tool (https://fair-checker.france-bioinformatique.fr/check) and potentially extend this with additional metrics.
3. **Analysis**: Analyze the evaluation results to derive statistics and visualizations.
4. **Conclusions**: Interpret the findings to answer the research questions and provide recommendations.

## Project Workflow
- **Initial Phase**: Project idea development, initial evaluation, and project proposal creation.
- **Data Collection**: Gather relevant datasets from Open Data Portals.
- **FAIRness Evaluation**: Use tool FAIR-checker to evaluate the datasets.
- **Analysis**: Analyze and visualize the results to identify trends and insights.
- **Final Phase**: Prepare a comprehensive final presentation summarizing the findings.

## Tools and Resources
- **FAIR-checker**: A tool to evaluate the FAIRness of data based on specific metrics. https://fair-checker.france-bioinformatique.fr/check

## Contributors
- **Matthäus Bulgarini**
- **Marcell Molnar**
