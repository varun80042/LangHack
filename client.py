import streamlit as st
import sys
import os

# Add the directory containing your main script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main graph creation function
from backup import create_graph

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Pharmaceutical Information Assistant",
        page_icon="💊",
        layout="wide"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .query-input {
        margin-bottom: 20px;
    }
    .answer-box {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 class='main-title'>🧬 Pharmaceutical Information Assistant</h1>", unsafe_allow_html=True)

    # Initialize or load chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create the graph
    try:
        app = create_graph()
    except Exception as e:
        st.error(f"Error initializing the pharmaceutical information system: {e}")
        return

    # Sidebar for additional information
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI-powered assistant provides pharmaceutical information 
        by searching through a local database and web sources when necessary.
        """)
        
        st.header("How It Works")
        st.write("""
        1. Enter your pharmaceutical query

        """)

    # Query input
    query = st.text_input(
        "Enter your pharmaceutical query", 
        placeholder="What would you like to know about medications?",
        key="query_input"
    )

    # Search button
    if st.button("Get Information", type="primary"):
        if query:
            # Clear previous results
            st.session_state.current_result = None
            
            # Spinner during processing
            with st.spinner('Searching for information...'):
                try:
                    # Invoke the graph with the query
                    result = app.invoke({"query": query})
                    
                    # Store the result
                    st.session_state.current_result = result
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": result.get('result', 'No information found.')
                    })
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Display current result
    if hasattr(st.session_state, 'current_result'):
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("### 📋 Answer")
        st.write(st.session_state.current_result.get('result', 'No information found.'))
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat History Expander
    with st.expander("📚 Chat History"):
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {chat['query']}")
                st.markdown(f"**A:** {chat['response']}")
                st.markdown("---")
        else:
            st.write("No previous queries.")

    # Clear History Button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()