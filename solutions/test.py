# How to download a public dataset from The Cancer Imaging Archive (TCIA)

# You can use the tcia-download-client or the tcia-api to download datasets.
# Here is an example using the tcia-api-client (requires installation):
# pip install tcia-api-client

#  To import functions related to NBIA for accessing our DICOM radiology data:
from tcia_utils import nbia
import requests
import json
import pandas as pd

# collection name
collection = "CPTAC-PDA"

# count patients for each modality
data = nbia.getModalityCounts(collection)
print(data)
     


# Count patients for each body part examined,
# return results as dataframe
df = nbia.getBodyPartCounts(collection, format = "df")
print(df.columns)

# rename headers and sort by PatientCount
df.rename(columns = {'criteria':'BodyPartExamined', 'Count':'PatientCount'}, inplace = True)
df.PatientCount = df.PatientCount.astype(int)
print(df.sort_values(by='PatientCount', ascending=False, ignore_index = True))
     
