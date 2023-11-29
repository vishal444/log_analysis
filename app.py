from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, UnstructuredODTLoader, CSVLoader, PyPDFLoader
import scipy
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/healthCheck', methods=['GET'])
def health_check():
    return "health check passed"


@app.route('/logs', methods=['POST'])
def process_logs():
    is_pdf = False
    is_txt = False
    is_csv = False
    is_odt = False
    api_key = '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'
    openai.api_key = api_key
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_key=api_key)

    if openai.api_key == api_key:
        print("API key is valid.")
    else:
        print("Invalid API key.")
    if 'logData' in request.form:
        logs = request.form['logData']
        log_data = [logs]
    if 'file' in request.files:
        uploaded_file = request.files['file']
        file_extension = uploaded_file.filename.rsplit('.', 1)[1].lower()

        if file_extension in {'pdf', 'txt', 'csv', 'odt'}:
            if file_extension == 'pdf':
                is_pdf = True
                uploaded_file.save('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.pdf')
            elif file_extension == 'txt':
                is_txt = True
                uploaded_file.save('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.txt')
            elif file_extension == 'csv':
                is_csv = True
                uploaded_file.save('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.csv')
            elif file_extension == 'odt':
                is_odt = True
                uploaded_file.save('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.odt')

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        if is_pdf:
            loader = PyPDFLoader('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.pdf')
        elif is_txt:
            loader = TextLoader('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.txt')
        elif is_csv:
            loader = CSVLoader('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.csv')
        elif is_odt:
            loader = UnstructuredODTLoader('/Users/vishaljose/PycharmProjects/logAnalysisBack/log.odt')

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        db = FAISS.from_documents(documents, embeddings)


        query = "When was kerala formed ?"
        docs = db.similarity_search(query)
        retriever = db.as_retriever()
        qa_stuff = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=True
        )
        query = "Where is kerala located. And answer in one sentence ?"
        response = qa_stuff.run(query)
    if log_data is not None:
        template_string = """
        Do log analysis for the following text
        ```{text}```
        """
        text = log_data
        prompt_template = ChatPromptTemplate.from_template(template_string)
        prompt = prompt_template.format_messages(
            text=text)
        response = llm(prompt)
        print(response)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)