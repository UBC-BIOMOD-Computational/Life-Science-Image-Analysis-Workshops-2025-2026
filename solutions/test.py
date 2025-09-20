# How to download a public dataset from The Cancer Imaging Archive (TCIA)

# You can use the tcia-download-client or the tcia-api to download datasets.
# Here is an example using the tcia-api-client (requires installation):
# pip install tcia-api-client

#  To import functions related to NBIA for accessing our DICOM radiology data:
from tcia_utils import nbia
import requests
import json
import io
import pandas as pd

# Use production endpoint instead of test
baseurl = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
query_endpoint = "/getPatientStudy"
query_parameters = "Collection=TCGA-GBM&PatientID=TCGA-02-0001"  # Use valid patient ID and remove spaces
_format = "csv"

cases_endpt = 'https://api.gdc.cancer.gov/cases'
collections = ['TCGA-BLCA']

# filters = {
#     "op": "in",
#     "content":{
#         "field": "project.project_id",
#         "value": collections
#         }
#     }

# fields = [
#     "project.project_id",
#     "submitter_id",
#     ]

# fields = ','.join(fields)

# expand = [ ## For the allowable values for this list, look under "mapping" at https://api.gdc.cancer.gov/cases/_mapping
#     "demographic",
#     "diagnoses",
#     "diagnoses.treatments",
#     "exposures",
#     "family_histories"
#     ]

# expand = ','.join(expand)

# params = {
#     "filters": json.dumps(filters),
#     "expand": expand,
#     "fields": fields,
#     "format": "TSV", ## This can be "JSON" too
#     "size": "10000", ## If you are re-using this for other projects, you may need to modify this and the "from" number.
#     "from":"0"
#     }

# response = requests.get(cases_endpt, params = params)

# output = response.content.decode('UTF-8')
# clinicalDf = pd.read_csv(io.StringIO(output), sep='\t')

# clinicalDf

# patientCollection = nbia.getPatient(collections[0])
# patients = pd.DataFrame(patientCollection, columns=['PatientId'])

# # create new dataframe from patients with only unique IDs of patients with imaging
# uniquePatients = pd.DataFrame(patients['PatientId'].unique(), columns=['PatientId'])

# # Rename the patient id column to match uniquePatients
# clinicalDf = clinicalDf.rename(columns={'submitter_id': 'PatientId'})

# # Merge the dataframes
# mergedClinical = uniquePatients.merge(clinicalDf, how='left', on='PatientId')

# # Drop columns with all NaN values from clinical data
# cleanClinical = mergedClinical.dropna(axis=1, how='all')

# # feel free to change this to other tissue types
# tissue_type = "lung"

# # Create dataframe for selected tissue type
# tissue_type_df = cleanClinical[cleanClinical['diagnoses.0.tissue_or_organ_of_origin'].str.contains(tissue_type, case=False, na=False)]

# project_ids = tissue_type_df['project.project_id'].unique().tolist()


# get list of all collections
collections_json = nbia.getCollections()
print(str(len(collections_json)) + " collections were found.")
collections = [item['Collection'] for item in collections_json]
print('Collections: ', collections)
collections = ['PROSTATE-DIAGNOSIS']

series_df = nbia.getSeries(collections[0], format="df")

print("Series DataFrame shape:", series_df.shape)
print(series_df.head())
try:
    series_uid = series_df['SeriesInstanceUID'].iloc[0]
    print(f"\nAttempting to download series: {series_uid}")
    
    # Download with more specific parameters
    nbia.downloadSeries(
        [series_uid], 
        number=1, 
        input_type="list",
        path="./downloads"  # Specify download directory
    )
    print("Download completed successfully!")
    
except Exception as e:
    print(f"Download failed: {e}")