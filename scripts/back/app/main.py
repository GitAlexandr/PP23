from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from langchain.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from translate import Translator
import os
import csv
import gradio as gr

app = FastAPI()

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="mistral",
        max_new_tokens = 1048,
        temperature = 0.3
    )
    return llm

import io
import os
import tempfile

def file_processing(file_content):
    if file_content is None:
        raise ValueError("Invalid file content: None")

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Load data from PDF
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    # Clean up the temporary file
    os.unlink(temp_file_path)

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen





def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # Load model for question generation
    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    Вы являетесь экспертом в создании вопросов на русском языке на основе материалов по документации.
    Ваша цель – подготовить специалиста к экзамену и тестам. Вопросов должно быть не менее 10.
    Вы делаете это, задавая вопросы с вопросительными словами на русском языке по тексту ниже:

    ------------
    {text}
    ------------

    Создавайте вопросы на русском языке, которые подготовят специалистов к тестам.
    Убедитесь, что не потеряете важную информацию.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    Вы являетесь экспертом в создании практических вопросов на русском языке на основе документации.
    Ваша цель – помочь специалисту пройти тест.
    Вы получили несколько практических вопросов в определенной степени: {existing_answer}.
    У вас есть возможность уточнить существующие вопросы или добавить новые.
    (только при необходимости) с дополнительным контекстом ниже.
    ------------
    {text}
    ------------

    Учитывая новый контекст, уточните исходные вопросы на русском языке.
    Если контекст бесполезен, предоставьте исходные вопросы.
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt=PROMPT_QUESTIONS,
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Load model for answer generation
    llm_answer_gen = load_llm()

    ques_list = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever())


    return answer_generation_chain, ques_list



def get_csv(file_content):
    answer_generation_chain, ques_list = llm_pipeline(file_content)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)

    output_file = os.path.join(base_folder, "QA.csv")

    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])
        for question in ques_list:
            answer = answer_generation_chain.run(question)
            translator = Translator(to_lang="ru")
            translated_question = translator.translate(question)
            translated_answer = translator.translate(answer)
            csv_writer.writerow([translated_question, translated_answer])

    return output_file

def generate_questions(file: UploadFile = File(...)):
    output_file = get_csv(file.file.name)  
    return FileResponse(output_file, filename="QA.csv")


@app.post("/generate_questions/")
def generate_questions(file: UploadFile = File(...)):
    file_content = file.file.read()

    # Ensure a valid file content is provided
    if not file_content:
        raise ValueError("Invalid file content")

    # Call llm_pipeline with the file content
    output_file = get_csv(file_content)

    return FileResponse(output_file, filename="QA.csv")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)