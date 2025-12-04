import streamlit as st
from src.semantic_search_engine import retrieve_data, generate_answers

# ---- Page setup ----
st.set_page_config(page_title="Semantic Search Demo", layout="wide")
st.title("🔍 KDF Search Engine")

# ---- Input area ----
# st.subheader("Enter your semantic search query")
st.markdown("""
<style>
/* Style the text input box */
div[data-baseweb="input"] input {
    font-size: 1.3rem !important;         /* Bigger text */
    font-family: 'Source Sans Pro', sans-serif !important;  /* Match Streamlit font */
    padding: 0.5rem 1rem !important;      /* Bigger box */
}

/* Style the Search button */
div.stButton > button {
    padding: 0.1rem 1rem !important;
    font-size: 1.3rem !important;
    font-family: 'Source Sans Pro', sans-serif !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])  # wider text input, narrower button
with col1:
    query = st.text_input(
        label="",  # required argument, even when collapsed
        label_visibility="collapsed",
        placeholder="Enter your semantic search query",
    )
with col2:
    search_clicked = st.button("Search", use_container_width=True)

if search_clicked: # query is bigger. Search on the right
    with st.spinner("🔍 Searching... please wait"):
        raw = retrieve_data(query)
        result = generate_answers(query, raw)
    if "error" in result:
        st.error(result["error"]) # add spinning wheel when
    else:
        col_left, col_right = st.columns([2, 2])

        # ---- Left column: Overall summary ----
        with col_left:
            st.subheader("🧠 Answer Summary") # make it bigger 50%
            st.info(result.get("answer", "No relevant data can be found."))
            name_top = result.get("name_top", "Unnamed Dataset")
            html = f"""
            <div style="
                padding:6px 10px;
                border-radius:6px;
                display:inline-block;
                font-size:1rem;             /* smaller text */
                color:#0A8A34;                 /* green text */
                text-decoration: underline;    /* underline text */
            ">
            📂 {name_top}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        # ---- Right column: Dataset details ----
        with col_right:
            st.subheader("Results")
            summary_data = result.get("supporting_datasets", [])
            if not summary_data:
                st.warning("No dataset summaries found.")
            else:
                for dataset in summary_data:
                    name_other = dataset.get('name', 'Unnamed Dataset')
                    html_other = f"""
                    <div style="
                        background-color:#E8F8EE;      /* light green background */
                        padding:6px 10px;
                        border-radius:6px;
                        display:inline-block;
                        font-size:1rem;             /* smaller text */
                        color:#0A8A34;                 /* green text */
                        text-decoration: underline;    /* underline text */
                    ">
                    📂 {name_other}
                    </div>
                    """

                    st.markdown(html_other, unsafe_allow_html=True)
                    # st.markdown(f"### 📂 {dataset.get('name', 'Unnamed Dataset')}")
                    st.markdown(f"**Summary:** {dataset.get('summary', 'No summary')}")
                    st.markdown( # add "citation from text.."
                        f"> *{dataset.get('quote', 'No representative quote available.')}*"
                    )
                    st.markdown("---")
