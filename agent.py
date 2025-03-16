import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqDataScienceAgent:
    
    def __init__(self, api_key=None, model_id="llama3-70b-8192"):
        """Initialize the Groq client using API key."""
        # Get API key from environment variable, Streamlit secrets, or directly
        try:
            self.api_key = api_key or os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
        except:
            self.api_key = None
        
        # API endpoint
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Store model ID
        self.model_id = model_id
        
        # Conversation history
        self.conversation_history = []
    
    def set_model(self, model_id):
        """Update the model being used."""
        self.model_id = model_id
    
    def ask(self, query, system_prompt=None):
        """Ask the model a question and get a response."""
        if not self.api_key:
            return "⚠️ API key not set. Please enter your Groq API key in the sidebar."
            
        default_system_prompt = """You are a helpful data science assistant. 
        You provide clear, accurate information about data analysis, statistics, machine learning, 
        and data visualization. When providing code, ensure it is correct, efficient, and well-documented.
        Focus on Python-based data science libraries like pandas, numpy, scikit-learn, matplotlib, 
        and seaborn. If asked about data, always suggest using pandas."""
        
        # Use custom system prompt if provided, otherwise use default
        system = system_prompt if system_prompt else default_system_prompt
        
        # Prepare messages in the OpenAI format that Groq uses
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        try:
            # Call Groq API
            with st.spinner(f"Thinking using {self.model_id}..."):
                start_time = time.time()
                response = requests.post(self.api_url, headers=headers, json=payload)
                end_time = time.time()
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Extract response
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                
                # Calculate response time
                response_time = end_time - start_time
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": answer})
                
                return answer, response_time
        
        except requests.exceptions.RequestException as e:
            return f"Error calling Groq API: {str(e)}\nResponse: {getattr(e.response, 'text', 'No response text')}", 0
        except Exception as e:
            return f"Error processing response: {str(e)}", 0
    
    def analyze_dataframe(self, df, question):
        """Get the model to analyze a pandas DataFrame."""
        # Create a summary of the DataFrame
        df_info = f"DataFrame info:\n"
        df_info += f"- Shape: {df.shape}\n"
        df_info += f"- Columns: {', '.join(df.columns.tolist())}\n"
        df_info += f"- Data types:\n{df.dtypes.to_string()}\n"
        df_info += f"- First 5 rows:\n{df.head().to_string()}\n"
        df_info += f"- Summary statistics:\n{df.describe().to_string()}\n"
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            df_info += f"- Missing values:\n{missing[missing > 0].to_string()}\n"
        
        # Ask about the DataFrame
        prompt = f"Here is information about a pandas DataFrame:\n\n{df_info}\n\nQuestion: {question}\n\nPlease analyze this information and answer the question. Include Python code when relevant."
        return self.ask(prompt)
    
    def suggest_visualization(self, df, columns=None):
        """Suggest visualizations for the given DataFrame."""
        # Create a summary of the DataFrame
        df_info = f"DataFrame info:\n"
        df_info += f"- Shape: {df.shape}\n"
        df_info += f"- Columns: {', '.join(df.columns.tolist())}\n"
        df_info += f"- Data types:\n{df.dtypes.to_string()}\n"
        
        if columns:
            col_info = "Specific columns to visualize: " + ", ".join(columns)
        else:
            col_info = "Consider all columns for visualization suggestions."
        
        prompt = f"{df_info}\n\n{col_info}\n\nPlease suggest 2-3 appropriate data visualizations for this dataset. For each visualization, explain why it's useful and provide the exact Python code using matplotlib or seaborn to create it."
        return self.ask(prompt)
    
    def explain_code(self, code):
        """Get an explanation of Python data science code."""
        prompt = f"Please explain the following Python data science code step by step:\n\n```python\n{code}\n```\n\nFocus on what each line does and why it's important. If there are potential improvements or optimizations, please mention them."
        return self.ask(prompt)
    
    def suggest_ml_approach(self, task_description, data_description):
        """Suggest machine learning approaches for a given task and data."""
        prompt = f"Task: {task_description}\n\nData description: {data_description}\n\nPlease suggest appropriate machine learning approaches for this task. For each approach, explain:\n1. Why it's suitable\n2. What preprocessing steps would be needed\n3. How to implement it using scikit-learn or another appropriate library\n4. How to evaluate the results"
        return self.ask(prompt)
        
    def generate_linkedin_post(self, code):
        """Generate a LinkedIn post from a code snippet explaining the technique and its applications."""
        prompt = f"""Analyze this Python code snippet:

```python
{code}
```

First, briefly explain what the code does in 2-3 sentences.

Then, create an engaging LinkedIn post that:
1. Highlights the key technique or pattern demonstrated in the code
2. Explains why this approach is useful for data scientists
3. Mentions a real-world application or use case
4. Includes 3-5 relevant hashtags
5. Is professional but conversational in tone
6. Has a hook that will get attention
7. Is under 1300 characters (LinkedIn limit)

Format the LinkedIn post as if it's ready to copy and paste."""
        
        return self.ask(prompt)
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        return "Conversation history cleared."

# Initialize Streamlit app
st.set_page_config(page_title="Groq Data Science Agent", page_icon="🚀", layout="wide")

# Initialize the agent
@st.cache_resource
def get_agent():
    return GroqDataScienceAgent()

agent = get_agent()

# Sidebar for configuration
st.sidebar.title("Configuration")

# API Key input
api_key = st.sidebar.text_input("Groq API Key", type="password", 
                              help="Enter your Groq API key. It will not be stored permanently.", 
                              value=os.getenv("GROQ_API_KEY", ""))
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    agent.api_key = api_key

# Model selection - Groq models
model_options = {
    "Llama 3 (70B)": "llama3-70b-8192",
    "Llama 3 (8B)": "llama3-8b-8192",
    "Mixtral (8x7B)": "mixtral-8x7b-32768",
    "Gemma (7B)": "gemma-7b-it"
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys()),
    index=0
)
agent.set_model(model_options[selected_model])
    
# Button to clear conversation history
if st.sidebar.button("Clear Conversation History"):
    st.session_state.messages = []
    agent.clear_history()

# Test connection button
if st.sidebar.button("Test API Connection"):
    with st.sidebar:
        with st.spinner("Testing connection..."):
            test_response, response_time = agent.ask("Hello, just testing the connection. Please respond with 'Connection successful'.")
            if "Connection successful" in test_response or "success" in test_response.lower():
                st.success(f"API connection successful! Response time: {response_time:.2f}s")
            else:
                st.error(f"API connection issue: {test_response}")

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["General Q&A", "Analyze Dataset", "Suggest Visualizations", "Explain Code", "Suggest ML Approach", "LinkedIn Post Generator"]
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
st.title("Groq Data Science Assistant 🚀")
st.caption(f"A lightweight agent for data science tasks powered by {selected_model} via Groq")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process and display user queries
def process_query(query, mode="General Q&A", **kwargs):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if mode == "General Q&A":
            response, response_time = agent.ask(query)
        elif mode == "Analyze Dataset":
            if "df" in kwargs:
                response, response_time = agent.analyze_dataframe(kwargs["df"], query)
            else:
                response, response_time = "No dataset available for analysis.", 0
        elif mode == "Suggest Visualizations":
            if "df" in kwargs and "columns" in kwargs:
                response, response_time = agent.suggest_visualization(kwargs["df"], kwargs["columns"])
            else:
                response, response_time = "Missing dataset or columns for visualization.", 0
        elif mode == "Explain Code":
            if "code" in kwargs:
                response, response_time = agent.explain_code(kwargs["code"])
            else:
                response, response_time = "No code provided for explanation.", 0
        elif mode == "Suggest ML Approach":
            if "task" in kwargs and "data" in kwargs:
                response, response_time = agent.suggest_ml_approach(kwargs["task"], kwargs["data"])
            else:
                response, response_time = "Missing task or data description.", 0
        elif mode == "LinkedIn Post Generator":
            if "code" in kwargs:
                response, response_time = agent.generate_linkedin_post(kwargs["code"])
            else:
                response, response_time = "No code provided for LinkedIn post generation.", 0
        else:
            response, response_time = "Invalid mode selected.", 0
        
        message_placeholder.markdown(response)
        
        # Show response time
        if response_time > 0:
            st.caption(f"Response time: {response_time:.2f}s")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    return response

# Handle different modes
if mode == "General Q&A":
    # Text input for general questions
    query = st.chat_input("Ask a data science question...")
    if query:
        process_query(query)
        
elif mode == "Analyze Dataset":
    # File uploader for CSV
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            
            # Display basic stats
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", df.shape)
            with col2:
                st.write("Missing Values:", df.isnull().sum().sum())
            
            # Analysis question
            query = st.chat_input("Ask a question about this dataset...")
            if query:
                process_query(query, mode="Analyze Dataset", df=df)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.info("Please upload a CSV file to analyze.")

elif mode == "Suggest Visualizations":
    # File uploader for CSV
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
            
            # Column selection
            columns = st.sidebar.multiselect("Select columns for visualization", df.columns.tolist())
            
            if st.button("Suggest Visualizations"):
                process_query("Suggest visualizations for these columns", 
                             mode="Suggest Visualizations", 
                             df=df, 
                             columns=columns)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.info("Please upload a CSV file and select columns.")

elif mode == "Explain Code":
    # Text area for code input
    code = st.text_area("Enter Python code to explain:", height=300)
    
    if st.button("Explain Code"):
        process_query("Explain this code", mode="Explain Code", code=code)

elif mode == "Suggest ML Approach":
    # Text inputs for task and data description
    task = st.text_area("Describe your machine learning task:", height=100)
    data = st.text_area("Describe your dataset:", height=100)
    
    if st.button("Suggest Approaches"):
        process_query("Suggest ML approaches", 
                     mode="Suggest ML Approach", 
                     task=task, 
                     data=data)

elif mode == "LinkedIn Post Generator":
    st.header("Generate LinkedIn Post from Code")
    st.markdown("Paste your Python code snippet and get a professionally crafted LinkedIn post explaining the technique and its applications.")
    
    # Text area for code input
    code = st.text_area("Enter Python code to convert to a LinkedIn post:", height=300, 
                        placeholder="Paste your Python code here...")
    
    # Generate button
    if st.button("Generate LinkedIn Post"):
        if not code:
            st.warning("Please enter some code to generate a LinkedIn post.")
        else:
            response = process_query("Generate LinkedIn post from code", 
                                    mode="LinkedIn Post Generator", 
                                    code=code)
            
            # Extract just the post part (skip the explanation at the beginning)
            parts = response.split("\n\n", 1)
            post_text = parts[1] if len(parts) > 1 else response
            
            # Add a text area for easy copying
            st.text_area("Copy this text:", value=post_text, height=200)
            st.info("Select and copy the text above to paste into LinkedIn")

# Footer
st.markdown("---")
st.caption("This agent uses Groq's API for ultra-fast LLM inference. Responses are generated in real-time.")
