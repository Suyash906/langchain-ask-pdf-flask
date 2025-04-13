from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms.openai import OpenAI


app = Flask(__name__)

@app.route('/ask-pdf', methods=['POST'])
def ask_pdf():
    try:
        # Load environment variables
        load_dotenv()

        # Get the PDF file and question from the request
        if 'pdf' not in request.files or 'question' not in request.form:
            return jsonify({"error": "PDF file and question are required"}), 400

        pdf_file = request.files['pdf']
        user_question = request.form['question']

        # Read the PDF file
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Perform similarity search
        answer = knowledge_base.similarity_search(user_question)

        # Load the LLM and QA chain
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        # Get the response
        with get_openai_callback() as cb:
            response = chain.run({
                "input_documents": answer,
                "question": user_question
            })
            print(cb)  # Log the API call cost

        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    # main()
    app.run(debug=True)