from io import BytesIO
import os
import asyncio
from docx import Document
import pandas as pd
from langchain_openai import ChatOpenAI
import streamlit as st
from crewai import Agent, Task, Crew
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.replicate import Replicate
from unstructured.partition.auto import partition

def load_data(file_path):
    
    return partition(file_path,strategy="hi_res")

def generate_text(data, llm, mode):
    Settings.chunk_size = 512
    documents = [{"text": elem.text if isinstance(elem, dict) else elem['text'], "metadata": {}} for elem in data]

    if mode == 'OpenAI':
        index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
        query_engine = index.as_query_engine()
        result = query_engine.query("Could you summarize the given context? Return your response which covers the key points of the text and does not miss anything important, please.")
        
    elif mode == 'llama':
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    
        index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
        query_engine = index.as_query_engine()
        result = query_engine.query("Could you summarize the given context? Return your response which covers the key points of the text and does not miss anything important, please.")
        
    if isinstance(result, dict) and 'text' in result:
        return result['text']
    else:
        return str(result)

def formatter(result, llm):
    agent_formatter = Agent(
        role='Summary Formatter',
        goal='Make the summary clear, concise, and well-structured without missing out on any important details',
        backstory='An expert in distilling information into readable formats.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_formatter = Task(
        description=f'''Format the given summary text to be clear and structured with headings and readable paragraphs.
                       The given summary is: {result}''',
        agent=agent_formatter,
        expected_output='''A well-structured summary with clear headings and organized content.
                           Nothing important should be missed out on from the summary'''
    )
    
    crew = Crew(
            agents=[agent_formatter],
            tasks=[task_formatter],
            verbose=2
        )
    
    return crew.kickoff()

def main():
    global mod
    mod = None

    st.header('Summary Generator') 
    
    # Initialize session state for storing summaries
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
        
    with st.sidebar:
        with st.form('OpenAI,llama2-70B'):
            model = st.radio('Choose Your LLM', ('OpenAI','Replicate/llama2-70B'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

    if api_key:
        if model == 'OpenAI':
            async def setup_OpenAI():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(model='gpt-4-turbo',temperature=0.6, max_tokens=2000,api_key=api_key)
                print("OpenAI Configured")
                return llm

            llm = asyncio.run(setup_OpenAI())
            mod = 'OpenAI'

        elif model == 'Replicate/llama2-70B':
            async def setup_llama():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                os.environ["REPLICATE_API_TOKEN"] = api_key
                llm = Replicate(
                        model="meta/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
                        is_chat_model=True,
                        additional_kwargs={"max_new_tokens": 512})
                print("Llama Configured")
                return llm

            llm = asyncio.run(setup_llama())
            mod = 'llama'

        upload_directory = "Saved Files"
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)

        # File uploader for multiple files

        uploaded_files = st.file_uploader("Choose files", type=None, accept_multiple_files=True)

        if uploaded_files and st.button("Process Files"):
            st.session_state.summaries = []  # Clear previous summaries
            for file in uploaded_files:
                with st.spinner(f"Generating summary for {file.name}..."):
                    # Save the file temporarily
                    temp_file_path = os.path.join(upload_directory, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Generate and format content
                    data = load_data(temp_file_path)
                    generated_content = generate_text(data, llm, mod)
                    formatted_content = formatter(generated_content, llm)

                    # Store the summary in session state
                    st.session_state.summaries.append({
                        'file_name': file.name,
                        'content': formatted_content
                    })

                    # Remove the temporary file
                    os.remove(temp_file_path)

        # Display all stored summaries
        for idx, summary in enumerate(st.session_state.summaries):
            st.subheader(f"Summary for {summary['file_name']}")
            st.markdown(summary['content'])

            # Create Word document
            doc = Document()
            doc.add_heading(f'Summary for {summary["file_name"]}', 0)
            doc.add_paragraph(summary['content'])

            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)

            # Download button for each summary
            st.download_button(
                label=f"Download Summary for {summary['file_name']} as Word Document",
                data=buffer,
                file_name=f"Summary_{summary['file_name']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"download_{idx}"  # Unique key for each download button
            )

            st.markdown("---")  # Add a separator between summaries

if __name__ == "__main__":
    main()
