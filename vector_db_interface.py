import streamlit as st
import requests

from interface_assets.responsive_styles import (
    apply_responsive_config,
    load_responsive_css,
    create_responsive_header,
    create_upload_container,
    close_upload_container,
    create_responsive_columns,
    display_files_responsive,
    create_responsive_button
)

# Replace with your actual FastAPI endpoint URL (must include /upload-pdf/)
API_URL = "http://0.0.0.0:10000/upload-pdf/"

# Responsive UI setup
apply_responsive_config()
load_responsive_css()
create_responsive_header("Vector Database Ingestion")

upload_col1, upload_col2, upload_col3 = create_responsive_columns([1, 3, 1])

with upload_col2:
    uploaded_files = st.file_uploader(
        "Upload the PDF You Want to Add to the Vector Database",
        accept_multiple_files=True,
        type="pdf"
    )

close_upload_container()

display_files_responsive(uploaded_files)

if uploaded_files:
    if create_responsive_button("🚀 Process Files"):
        with st.spinner("Processing files..."):
            results = []
            for uploaded_file in uploaded_files:
                try:
                    file_bytes = uploaded_file.read()  # Read file bytes properly
                    files = {"file": (uploaded_file.name, file_bytes, "application/pdf")}
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()  # Raise HTTP errors if any
                    out = response.json()
                    status = out.get("status", "Unknown")
                    message = out.get("message", "")
                    results.append(f"{uploaded_file.name}: {status} - {message}")
                except requests.exceptions.RequestException as e:
                    results.append(f"{uploaded_file.name}: Request failed - {e}")
                except Exception as e:
                    results.append(f"{uploaded_file.name}: Error - {e}")
            st.success(f"✅ Successfully sent {len(uploaded_files)} file(s) to the vector database API.")
            for res in results:
                st.write(res)
