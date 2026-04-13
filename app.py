import streamlit as st
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate


# --- UI Config ---
st.set_page_config(page_title="BLS Historical RAG", page_icon="📈")
st.title("🏛️ BLS Occupational Analyst (1949-2024)")

# --- Secure API Keys ---
# Locally: uses .streamlit/secrets.toml
# Cloud: uses the "Secrets" dashboard
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    
except KeyError:
    st.error("API Keys missing! Please set them in Streamlit Secrets.")
    st.stop()

# --- Initialize Engines ---
@st.cache_resource # Keeps connection alive without reloading every click
def init_rag():
    embeddings = JinaEmbeddings(
        jina_api_key=JINA_API_KEY, 
        model_name="jina-embeddings-v3"
    )
    
    vector_db = PineconeVectorStore(
        index_name="llama-text-embed-v2-index", 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.1, 
        groq_api_key=GROQ_API_KEY
    )
    
    return vector_db, llm

vector_db, llm = init_rag()

# --- System Prompt ---
system_prompt = ("""
    "Role: You are an expert People Analytics Consultant and Labor Historian specializing in the BLS Occupational Outlook Handbooks (OOH) from 1949 to 2024."
    "Task: Your goal is to extract, analyze, and synthesize workforce intelligence to identify long-term patterns in job roles, skills, and economic shifts."
    "Strict Grounding: Base your answers ONLY on the provided PDF context. If the information is not present in the documents, state:I do not have sufficient historical data in the uploaded handbooks to answer this."
    "Handling Time-Series Data: When asked about a job role, always specify the Handbook Year you are referencing. If multiple years are available, provide a comparative timeline."
    "Key Pillars of Analysis: For every job role, prioritize extracting:Entry Barriers: Educational and training requirements.Technological Drivers: Mention of new tools (e.g., calculators, computers, AI).Economic Outlook: Projected growth or decline as stated by the BLS at that time."
    "Output Structure: Use Markdown tables for comparisons. Use bullet points for Key Insights and Skill Gaps."
    "Constraint: Never use outside general knowledge to fill in the blanks of a specific handbook's data."
    "Linguistic Sensitivity: Recognize that terminology changes. (e.g., Draftsmen in 1950 vs CAD Operators in 2020). Treat these as a single professional evolution."
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
    """
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- Logic ---
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vector_db.as_retriever(search_kwargs={"k": 5}), 
    question_answer_chain
)

# --- App UI ---
user_input = st.text_input("Ask a question about workforce history:")

if user_input:
    with st.spinner("Analyzing historical records..."):
        response = rag_chain.invoke({"input": user_input})
        
        st.markdown(f"### 🤖 Answer:\n{response['answer']}")
        
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown PDF')}")
                st.caption(doc.page_content[:300] + "...")