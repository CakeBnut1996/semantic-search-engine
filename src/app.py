import streamlit as st
from semantic_search_engine import retrieve_data, generate_answers

# ---- Page setup ----
st.set_page_config(page_title="Semantic Search Demo", layout="wide")
st.title("🔍 Semantic Search Dashboard")

# ---- Input area ----
st.subheader("Enter your semantic search query")

# ---- Default / static behavior ----
query = st.text_input("Ask something like 'What research areas are needed to evaluate the feasibility of marine biofuels?'")
if st.button("Search"):
    with st.spinner("Searching..."):
        raw = retrieve_data(query)
        result = generate_answers(query, raw)
    if "error" in result:
        st.error(result["error"])
    else:
        # ---- Layout: Left 20%, Right 80% ----
        col_left, col_right = st.columns([1, 3])

        # ---- Left column: Overall summary ----
        with col_left:
            st.subheader("🧠 Answer Summary")
            st.info(result.get("answer", "No relevant data can be found."))
            name_top = result.get("name_top", "Unnamed Dataset")
            html = f"""
            <div style="
                padding:6px 10px;
                border-radius:6px;
                display:inline-block;
                font-size:0.85rem;             /* smaller text */
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
                    st.markdown(
                        f"> *{dataset.get('quote', 'No representative quote available.')}*"
                    )
                    st.markdown("---")
