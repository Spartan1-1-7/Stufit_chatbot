import requests

# Replace with the actual public API URL from ngrok, always ending in /upload-pdf/
API_URL = "http://0.0.0.0:10000/upload-pdf/"  # <-- Change if needed

# Replace with the path to your PDF file
PDF_FILE_PATH = "ingestion_source/Kaplan and Sadock's Synopsis of Psychiatry_ Behavioral -- Benjamin J_ Sadock, Virginia A_ Sadock, Dr_ Pedro Ruiz MD -- Paperback, 2014 -- Lippincott -- 9781451194340 -- c285583cd25b6a61f55458cce183a005 -- Anna’s Archive.pdf"

with open(PDF_FILE_PATH, "rb") as file:
    files = {"file": (PDF_FILE_PATH.split('/')[-1], file, "application/pdf")}
    response = requests.post(API_URL, files=files)
    print("Status Code:", response.status_code)
    try:
        print("Response:", response.json())
    except Exception as e:
        print("Non-JSON Response:", response.text)
        print("Error:", e)