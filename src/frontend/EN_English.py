import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from datetime import datetime
from PIL import Image
# from src.settings import user_log_collection
from src.frontend.utils import add_logo, on_button_click, get_avatars
from src.core.en_azure.azure_chatbot import AzureChatbot
# from src.core.en.pipeline import LlamaIndexChatbot
import os
from distutils.util import strtobool
# import nest_asyncio

# async
# nest_asyncio.apply()
DEBUG_MODE = False #strtobool(os.getenv("DEBUG_MODE"))
DEFAULT_MODEL = "Azure" #os.getenv("DEFAULT_MODEL")
image_directory = "./assets/images/mil_logo.png"
image = Image.open(image_directory)
avatars = get_avatars()
st.set_page_config(
    page_title="HKIRD eAOM Chatbot",
    page_icon=image)

# make model an option and put COST_PER_TOKEN under states
COSTING_MAP = {
    "gpt-3.5-turbo": 0.000002,
    "gpt-4": 0.00006
}

MODEL_FACTORY = {
    "Azure": AzureChatbot,
    "LlamaIndex": AzureChatbot
}

print("Clicked English page")

#st.image("./assets/images/manulife_logo_cropped.png", width=220)

st.title("HKIRD Intelligent Form Search 🤖")
session_id = get_script_run_ctx().session_id

add_logo()

if "messages" not in st.session_state:
    print("no messages detected, initializing states")
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.cost_of_response = 0
    # Assume default chatbot is azure
    st.session_state.model = "Azure"
    #st.session_state.chatbot = MODEL_FACTORY["Azure"]
    st.session_state.llm_model = "gpt-4"
    

# code block for chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=avatars[message["role"]]):
        st.markdown(message["content"], unsafe_allow_html=True)
        # printing out citations if exists
        if "citations" in message.keys():
            print("citations detected")
            # expander version
            
            for index, message_citation in enumerate(message["citations"][:message["max_citations"]]):
                with st.expander(f"[{index+1}] {message_citation['filepath']}"):
                    st.markdown("""
                    
                    **Form URL:** {}
                                
                    **Form Description:** {}
                                
                    """.format(message_citation["url"], message_citation["content"]))

            # citation link version
            # for message_citation in message["citations"]:
            #     mention(label=message_citation["filepath"], url=message_citation["url"])
            #message.pop("citations", None)


# code block for user input
if user_prompt := st.chat_input("Ask a question! e.g Can you provide me with a form for claiming my recent hospital stay that cost less than HKD 3000?"):
    user_messages = {"role": "user", "content": user_prompt}
    st.session_state.messages.append(user_messages)

    # current_time = datetime.now()
    # user_log_collection.insert_one(
    #     {'session_id': session_id, 
    #      'messages': user_messages, 
    #      'model': MODEL_FACTORY[st.session_state.model].get_name(), 
    #      'unixtimestamp': datetime.timestamp(current_time)*1000,
    #      'datetime':current_time}
    # )

    with st.chat_message("user", avatar=avatars["user"]):
        st.markdown(user_prompt)

    # create an instance of ChatResponseGenerator
    response_generator = st.session_state.chatbot(
        conversation=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
    )

    # generate responses
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        message_placeholder = st.empty()
        full_response = ""

        # streaming mode in a for loop
        for partial_response in response_generator.response_stream():
            full_response += partial_response if partial_response is not None else ""
            message_placeholder.markdown(full_response + "▌")

        full_response, max_citations = response_generator.preprocess_response(full_response)
        message_placeholder.markdown(full_response, unsafe_allow_html=True)

        assistant_messages = {"role": "assistant", "content": full_response}

        # citation handling
        message_citations = response_generator.citations

        if len(message_citations) > 0:
            assistant_messages["citations"] = response_generator.citations
            assistant_messages["max_citations"] = max_citations
            print("citations detected")
            # expander version
            for index, message_citation in enumerate(message_citations[:max_citations]):
                with st.expander(f"[{index+1}] {message_citation['filepath']}"):
                    st.markdown("""
                    
                    **Form URL:** {}
                    
                    **Form Description:** {}
                                
                    """.format(message_citation["url"], message_citation["content"]))

            # citation link version
            # for message_citation in message_citations:
            #     mention(label=message_citation["filepath"], url=message_citation["url"])
            
    # current_time = datetime.now()
    # user_log_collection.insert_one(
    #     {'session_id': session_id, 
    #      'messages': assistant_messages, 
    #      'model': MODEL_FACTORY[st.session_state.model].get_name(), 
    #      'unixtimestamp': datetime.timestamp(current_time)*1000,
    #      'datetime': current_time}
    # )

    st.session_state.messages.append(assistant_messages)
    st.session_state.total_tokens += 0#num_tokens_from_messages(st.session_state.messages, st.session_state.model)
    st.session_state.cost_of_response = 0 #st.session_state.total_tokens * COSTING_MAP[st.session_state.llm_model]

with st.sidebar:
    if DEBUG_MODE:
        st.title("Chatbot Selection:")
        #st.markdown("""---""")
        st.session_state.model = st.radio("Select Chatbot for debugging 👇",
            ["Azure", "LlamaIndex"], on_change=on_button_click)
        st.title("Session Usage Stats:")
        #st.markdown("""---""")
        st.session_state.chatbot = MODEL_FACTORY[st.session_state.model]
        st.write("Selected model: ", st.session_state.llm_model)
        st.write("Total tokens used [TBD]: ", st.session_state.total_tokens)
        st.write("Total cost of request [TBD]: ${:.8f}".format(st.session_state.cost_of_response))
        # Display the button with custom color
        st.button("Clear Chat History and Tokens", on_click=on_button_click)
        st.markdown("""
        ---
        ### To be Developed [TBD]
        - Add token calculator incl. retrieved contexts
        - Add pdf viewer for pdfs/mds served in blob storage

        """)
    else:
        st.session_state.chatbot = MODEL_FACTORY[DEFAULT_MODEL]
        st.button("Clear Chat History and Tokens", on_click=on_button_click)
        st.markdown("""
        
        ### Introduction
        - This is a Chatbot developed by HK CIO Office Innovation R&D Team (HKIRD) for internal experimentation purposes
        - The source document for this Chatbot is the English eAOM document

        ### How to use
        - Type in your question in the chatbot and press `Enter`. `Shift` `+` `Enter` for newline
        - This Chatbot is programmed to only respond to queries related to the eAOM
        - There is ~ 2700 word limit per conversation, shall it be exceeded, just click the **Clear Chat History and Tokens** button

        ### Disclaimer 
        - Chatbot source document scope is Chapter 3.2.1 and Chapter 5 of English eAOM Underwriting
        - Responses of this Chatbot are for internal testing purposes only
        - Chinese version is still WIP as focus of this PoC is the English version
        """)
    # Create download button    