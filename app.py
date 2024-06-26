from io import BytesIO
import os
import asyncio
from docx import Document
import streamlit as st
from crewai import Agent, Task, Crew
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.replicate import Replicate

def generate_text(llm):
    
    Settings.chunk_size = 512

    if mod == 'OpenAI':
        documents = SimpleDirectoryReader('Saved Files').load_data()
        index = VectorStoreIndex.from_documents(documents,transformations=[SentenceSplitter(chunk_size=512)])
        query_engine = index.as_query_engine()
        result = query_engine.query("Could you summarize the given context? Return your response which covers the key points of the text and does not miss anything important, please.")
        
    elif mod == 'llama':
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    
        documents = SimpleDirectoryReader('Saved Files').load_data()
        index = VectorStoreIndex.from_documents(documents,transformations=[SentenceSplitter(chunk_size=512)])
        query_engine = index.as_query_engine()
        result = query_engine.query("Could you summarize the given context? Return your response which covers the key points of the text and does not miss anything important, please.")
    
    
    # Assuming result is a response object with text content
    if isinstance(result, dict) and 'text' in result:
        return result['text']  # Assuming 'text' is the key for the content
    else:
        return str(result)  # Convert to string if direct content

def main():
    global mod
    mod = None

    st.header('Summary Generator')
        
    with st.sidebar:
        with st.form('Gemini/OpenAI/Groq'):
            model = st.radio('Choose Your LLM', ('OpenAI','Replicate/llama2-70b'))
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


            llm = asyncio.run(setup_OpenAI())
            mod = 'OpenAI'


        elif model == 'Replicate/llama2-70b':
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

            
        if st.button("Generate Content"):
            with st.spinner("Generating content..."):
                generated_content = generate_text(llm)

                st.markdown(generated_content)

                doc = Document()

                # Option to download content as a Word document
                doc.add_heading('Summary', 0)
                doc.add_paragraph(generated_content)

                buffer = BytesIO()
                doc.save(buffer)
                buffer.seek(0)

                st.download_button(
                    label="Download as Word Document",
                    data=buffer,
                    file_name=f"Summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
if __name__ == "__main__":
    main()
