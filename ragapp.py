import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


st.title("📚 college knowledge assistant")

chunks=[
    "ABC Institute offers courses in Artificial Intelligence,Data Science, Python, Machine Learning and Web Development.",
        "The Artificial Intelligence course duration is 6 months.",
        "The Data Science course duration is 8 months.",
        "Python is taught during the first two months of the Artificial Intelligence course.",
        "The Machine Learning module is taught during months three and four.",
        "The AI course includes Python, Machine Learning,Deep Learning, NLP and Generative AI."
        "Students must complete a final project to receive the course certificate.",
    
        "Classes are conducted from Monday to Friday.",
    
        "The institute provides both online and offline classes."
    
    ]
    
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    llm = pipeline(
        "text-generation",
         model="QWen/Qwen2.5-0.5B-Instruct" 
    )
    return embedding_model,llm

embedding_model,llm = load_models()

chunk_embeddings = embedding_model.encode(
    chunks
)
question = st.text_input(
    "ask a question about the institute:"
)
if question:
    question_embedding = embedding_model.encode(
        question
    )

    scores = cosine_similarity(
        [question_embedding],
        chunk_embeddings

    )[0]
    top_k = 3
    top_indices = scores.argsort()[-top_k:][::-1]

    relevant_chunks=[
        chunks[i]
        for i in top_indices
    ]
    context="\n".join(relevant_chunks)

    messages=[
    {
        "role":"system",
        "content":"you are a helpful assistant.answer only using the provided context."
    },
    {
        "role":"user",
        "content":f"""context:{context}
        Question:{question}
    """
    }
    ]
    result = llm(
        messages,
        max_new_tokens=100,
        do_sample=False)
    
    st.subheader("answer")

    st.write(
        result[0]["generated_text"][-1]["content"]
    )
    with st.expander("retrieved context"):
        for chunk in relevant_chunks:
            st.write(chunk)


