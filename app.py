import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import streamlit as st
import traceback
import pandas as pd
from langchain.callbacks import get_openai_callback
from utils import parse_file, get_table_data, RESPONSE_JSON

load_dotenv()

# Initialize ChatOpenAI with GPT-4o mini
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=4000,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.2
)

# This is an LLMChain to create 10-20 multiple choice questions from a given piece of text.
template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions for grade {grade} students in {tone} tone.
Make sure that questions are not repeated and check all the questions to be conforming to the text as well.
Make sure to format your response like the RESPONSE_JSON below and use it as a guide.\
Ensure to make the {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(template)
])

quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_prompt,
    output_key="quiz",
    verbose=True
)

# This is an LLMChain to evaluate the multiple choice questions created by the above chain
template = """You are an expert english grammarian and writer. Given a multiple choice quiz for {grade} grade students.\
You need to evaluate complexity of the questions and give a complete analysis of the quiz if the students 
will be able to understand the questions and answer them. Only use at max 50 words for complexity analysis.
If quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the students abilities. 
Quiz MCQs:
{quiz}
Critique from an expert english writer of the above quiz:"""

review_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(template)
])

review_chain = LLMChain(
    llm=llm,
    prompt=review_prompt,
    output_key="review",
    verbose=True
)

# This is the overall chain where we run these two chains in sequence.
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "grade", "tone", "response_json"],
    # Here we return multiple variables
    output_variables=["quiz", "review"],
    verbose=True,
)

st.set_page_config(
    page_title="AI Quiz Generator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Quiz Generator")
st.markdown("Generate multiple choice questions from your educational content")

# Create a form using st.form
with st.form("user_inputs"):
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF or text file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )

    # Input fields in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        mcq_count = st.number_input(
            "Number of MCQs",
            min_value=3,
            max_value=20,
            value=5,
            help="Choose between 3-20 questions"
        )
    with col2:
        grade = st.number_input(
            "Grade Level",
            min_value=1,
            max_value=10,
            value=5,
            help="Select grade level (1-10)"
        )
    with col3:
        tone = st.text_input(
            "Quiz Tone",
            value="simple",
            max_chars=100,
            placeholder="simple",
            help="e.g., simple, professional, friendly"
        )

    button = st.form_submit_button("Generate Quiz", use_container_width=True)

# Check if the button is clicked and all fields have inputs
if button and uploaded_file is not None and mcq_count and grade and tone:
    with st.spinner("Loading..."):
        try:
            text = parse_file(uploaded_file)

            # count tokens and cost of api call
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "grade": grade,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON),
                    }
                )
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")
        else:
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

            if isinstance(response, dict):
                # Extract quiz data from the response
                quiz = response.get("quiz", None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        
                        # Display quiz in a more modern way
                        st.subheader("Generated Quiz")
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=False
                        )
                        
                        # Display the review in an expander
                        with st.expander("Expert Review", expanded=False):
                            st.write(response["review"])
                    else:
                        st.error("Error in table data")
            else:
                st.write(response)
                