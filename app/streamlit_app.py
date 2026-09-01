import streamlit as st
import requests

st.set_page_config(page_title="Semantic Question Deduplication Engine", layout="centered")

st.title("🔍 Semantic Question Search & Deduplication")
st.markdown("Search over **10,000+ Quora questions** using Two-Stage Vector Search + Cross-Encoder Re-Ranking.")

query = st.text_input("Enter your question:", value="How can I start learning machine learning?")
top_k = st.slider("Top matches to retrieve", min_value=1, max_value=10, value=3)
enable_rerank = st.checkbox("Enable Cross-Encoder Re-Ranking", value=True)

if st.button("Find Similar Questions"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching vector index..."):
            try:
                # Increased timeout to 30 seconds for cold-starts
                response = requests.post(
                    "http://127.0.0.1:8000/search",
                    json={"question": query, "top_k": top_k, "rerank": enable_rerank},
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [])
                    
                    if not matches:
                        st.info("No matches found.")
                    else:
                        st.subheader("Top Matches:")
                        for idx, match in enumerate(matches, 1):
                            st.markdown(f"### {idx}. {match['question']}")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Bi-Encoder Score", f"{match.get('score', 0):.4f}")
                            
                            rerank_score = match.get("rerank_score")
                            if rerank_score is not None:
                                col2.metric("Cross-Encoder Score", f"{rerank_score:.4f}")
                            
                            st.divider()
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server: {e}")