import streamlit as st
from pathlib import Path
import google.generativeai as genai
from api_key import api_key

# Configure the generative AI model
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Finding Report: Document all observed anomalies or signs of diseases. Clearly articulate these findings in a structured format.
3. Recommendation and Next Steps: Based on your analysis, suggest potential next steps, including tests and treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.
2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are 'unable to be determined based on the provided image.'
3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with the Doctor before making any decisions."

4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.

Please provide me an output response with these 4 headings: Detailed Analysis, Finding Report, Recommendation and Next Steps, Treatment Suggestions.
"""

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Streamlit page configuration
st.set_page_config(page_title="Vital Image Analytics", page_icon=":robot:")
st.title("AI Disease Prediction from Medical Images")

# Instructions sidebar
st.sidebar.header("Instructions")
st.sidebar.write("Upload a medical image to get an analysis. Click 'Generate the Analysis' to start.")

# File uploader for medical images
uploaded_file = st.file_uploader("Upload the medical image for analysis", type=["png", "jpeg", "jpg"])

if uploaded_file:
    st.image(uploaded_file, width=300, caption="Uploaded Medical Image")

# Generate analysis button
submit_button = st.button("Generate the Analysis")

if submit_button:
    if uploaded_file is None:
        st.error("Please upload a valid medical image (PNG, JPEG, JPG).")
    else:
        with st.spinner("Generating analysis..."):
            image_data = uploaded_file.getvalue()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            prompt_parts = [image_parts[0], system_prompt]

            response = model.generate_content(prompt_parts)
            if response:
                st.title("Here is the analysis based on your images:")

                output_text = response.text.replace("**", "")  
                
                try:
                    detailed_analysis = output_text.split("Finding Report:")[0]
                    finding_report = output_text.split("Finding Report:")[1].split("Recommendation and Next Steps:")[0]
                    recommendation_next_steps = output_text.split("Recommendation and Next Steps:")[1].split("Treatment Suggestions:")[0]
                    treatment_suggestions = output_text.split("Treatment Suggestions:")[1]
                except IndexError as e:
                    st.error(f"Error processing analysis response: {e}")
                
                st.header("Detailed Analysis")
                st.write(detailed_analysis.strip())

                st.header("Finding Report")
                st.write(finding_report.strip())

                st.header("Recommendation and Next Steps")
                st.write(recommendation_next_steps.strip())

                st.header("Treatment Suggestions")
                st.write(treatment_suggestions.strip())

                st.download_button("Download Analysis", output_text, "analysis.txt")

# Feedback section
feedback = st.text_area("We appreciate your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
