import streamlit as st
from assistant import chat
import os
import re

def reset_state():
    # Reset the text input by clearing session state
    for key in st.session_state.keys():
        del st.session_state[key]

def parse_time_to_seconds(timestr):
    # Regular expression to find hours, minutes, and seconds
    time_pattern = re.compile(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?')
    match = time_pattern.search(timestr)
    
    # Extract hours, minutes, and seconds
    hours, minutes, seconds = match.groups(default='0')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return total_seconds

def main():
    st.title("Virtual Teaching Assistant")

    # PDF Upload section
    pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if pdf_file is not None:
        # To read file as bytes:
        bytes_data = pdf_file.getvalue()
        st.write("Uploaded PDF file.")
        # Optionally, save the file to disk
        with open(os.path.join('uploaded_files', pdf_file.name), 'wb') as f:
            f.write(bytes_data)
        st.success("File Saved Successfully")

    # Initializing session state variables if they are not already initialized
    if 'query' not in st.session_state:
        st.session_state['query'] = ""

    query_input = st.text_input("Enter your question:", key="query")

    if st.button("Ask"):
        response = chat(query_input)  # Assuming response is a dict with potential keys 'answer' and 'context'
        print(response)
        print(type(response))
        if type(response) == str:
            st.write(response)
            return
        answer = response.get('answer', 'No answer provided.')
        st.write("Answer:", answer)

        if 'context' in response and isinstance(response['context'], list):
            for doc in response['context']:
                print(doc.metadata)
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    if 'image_path' in doc.metadata:
                        image_path = doc.metadata['image_path']
                        st.image(image_path)
                    if 'video' in doc.metadata:
                        video_url = doc.metadata.get('video', '#')
                        time_part = re.search(r'\?t=([0-9hms]+)', video_url)
                        if time_part:
                            start_seconds = parse_time_to_seconds(time_part.group(1))
                            video_id = video_url.split('/')[-1].split('?')[0]
                            embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}&autoplay=0&rel=0"
                            st.markdown(
                                f'<iframe width="560" height="315" src="{embed_url}" frameborder="0"; encrypted-media" allowfullscreen></iframe>',
                                unsafe_allow_html=True
                            )
                            print(embed_url)
                break


    # Reset Button
    if st.button("Reset Chat", on_click=reset_state):
        pass

if __name__ == "__main__":
    main()