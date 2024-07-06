from io import BytesIO
import os
import asyncio
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
from crewai import Agent, Task, Crew
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
import nest_asyncio
nest_asyncio.apply()

def init_parser(api_key):
    return(LlamaParse(api_key=api_key,result_type="markdown", verbose=True))

def generate_text(parser, llm, file_path):
    Settings.chunk_size = 512
    
    if mod == 'Gemini':
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    
    documents = parser.load_data(file_path)
    index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
    query_engine = index.as_query_engine()
    result = query_engine.query("Could you very concisely summarize the given content? Return your response which covers the key points of the text and does not miss anything important, please.")
    
    return result['text'] if isinstance(result, dict) and 'text' in result else str(result)

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

async def process_file(file, parser, llm, upload_directory):
    temp_file_path = os.path.join(upload_directory, file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    
    generated_content = generate_text(parser, llm, temp_file_path)
    formatted_content = formatter(generated_content, llm)
    
    os.remove(temp_file_path)
    
    return {
        'File Name': file.name,
        'Summary': formatted_content
    }

async def process_all_files(files, parser, llm, upload_directory):
    tasks = [process_file(file, parser, llm, upload_directory) for file in files]
    return await asyncio.gather(*tasks)

def main():
    global mod
    mod = None

    st.header('Summary Generator') 
    
    # Initialize session state for storing summaries
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
        
    with st.sidebar:
        with st.form('OpenAI,Gemini'):
            model = st.radio('Choose Your LLM', ('OpenAI','Gemini'))
            api_key = st.text_input(f'Enter your API key', type="password")
            llamaindex_api_key = st.text_input(f'Enter your llamaParse API key', type="password")
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

        elif model == 'Gemini':
            async def setup_gemini():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                print("Gemini Configured")
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'

        parser = init_parser(llamaindex_api_key)
        
        upload_directory = "Saved Files"
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)

        uploaded_files = st.file_uploader("Choose files", type=None, accept_multiple_files=True)

        if uploaded_files and st.button("Process Files"):
            with st.spinner("Generating summaries for all files..."):
                # Use asyncio.run to properly execute the asynchronous function
                summaries = asyncio.run(process_all_files(uploaded_files, parser, llm, upload_directory))
                
                if summaries and isinstance(summaries, list) and all(isinstance(item, dict) for item in summaries):

                # Convert summaries to DataFrame for table display
                    df = pd.DataFrame(summaries)

                    # Display summaries in a table
                    st.table(df)
                    
                    # Create a single Word document with all summaries
                    doc = Document()
                    for summary in summaries:
                        doc.add_heading(f'Summary for {summary["File Name"]}', level=1)
                        doc.add_paragraph(summary['Summary'])
                        doc.add_page_break()
                    
                    buffer = BytesIO()
                    doc.save(buffer)
                    buffer.seek(0)
                    
                    # Download button for all summaries
                    st.download_button(
                        label="Download All Summaries as Word Document",
                        data=buffer,
                        file_name="All_Summaries.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()
