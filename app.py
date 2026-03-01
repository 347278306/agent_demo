import time

import streamlit as st

from agent.react_agent import ReactAgent

st.title("智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for messages in st.session_state["messages"]:
    st.chat_message(messages["role"]).write(messages["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中。。。"):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write(capture(res_stream, response_messages))
        st.session_state["messages"].append({"role": "assistant", "content": response_messages[-1]})

        st.rerun()
