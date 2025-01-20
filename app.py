# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:06:15 2025

@author: DELL
"""

import pickle
import streamlit as st
import helper

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Main function for Streamlit app
def main():
    st.set_page_config(page_title='Duplicate Question Checker', page_icon="❓", layout="centered")
    
    # App title and description
    st.title("❓ Duplicate Question Checker")
    st.subheader("Check if two questions are similar or duplicates")
    st.write("This tool leverages natural language processing to evaluate the similarity between two questions.")

    # Input fields
    q1 = st.text_area("Question 1", placeholder="Type the first question here...")
    q2 = st.text_area("Question 2", placeholder="Type the second question here...")
    
    # Initialize result variable
    answer = ''
    
    # Button to trigger prediction
    if st.button("Find"):
        query = helper.query_function(q1, q2)
        result = loaded_model.predict(query)
        if result[0] == 1:
            answer = 'Duplicate'
        else:
            answer = 'Not Duplicate'
        
        st.success(answer)  # Display result after prediction

# Entry point for the Streamlit app
if __name__ == '__main__':
    main()
