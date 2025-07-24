import streamlit as st
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

# Apply responsive configuration
apply_responsive_config()

# Load responsive CSS
load_responsive_css()

# Create responsive header
create_responsive_header("📚 Vector Database Ingestion")

# Main upload section with responsive container
# create_upload_container()

# Create responsive columns for the upload area
upload_col1, upload_col2, upload_col3 = create_responsive_columns([1, 3, 1])

with upload_col2:
    uploaded_files = st.file_uploader(
        "Upload the PDF You Want to Add to the vector Database",
        accept_multiple_files=True, 
        type="pdf"
    )

close_upload_container()

# Display uploaded files in responsive layout
display_files_responsive(uploaded_files)

# Process button with responsive positioning
if uploaded_files:
    if create_responsive_button("🚀 Process Files"):
        with st.spinner("Processing files..."):
            # Add your file processing logic here
            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")
            # st.balloons()