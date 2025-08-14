import streamlit as st

st.title("Streamlit Demo by Bipul")

st.header("Header for the App")

st.subheader("Sub-Header of the app")

st.text("This is a example Text")

st.text_input("Enter your Name")
st.text_area("Enter the Prompt")

#Checkbox
#Checkbox
if st.checkbox("Select/Unselect"):
    st.text("User selected the checkbox")
else:
    st.text("User has not selected the checkbox")

st.success("Success")
st.warning("Warning")
st.info("Information")
st.error("Error")

#Radio Button
state = st.radio("What is your favorite color?" , ("Red","Green","Blue"))

if state == 'Green':
    st.success("That is my favorite color as well")

#SelectBox
occupation = st.selectbox("What do you do?" , ["Student","Vlogger","Engineer"])
st.text(f"Selected option is {occupation}")

#Button
if st.button("Example Button"):
    st.success("You clicked it")