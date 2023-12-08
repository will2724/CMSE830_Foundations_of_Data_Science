import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
#########################################################################################################
with st.sidebar: 
    selected = option_menu(
        menu_title="Menu",
        options=['EDA🚀', 'Salary Analysis', 'Salary Prediction🎱', 'Conclusion','About Me']
    )

if selected == 'EDA🚀':
    st.title('Exploratior Data Analysis')

if selected == 'Salary Analysis':
    st.title('Salary Analysis')

if selected == 'Salary Prediction🎱':
    st.title('')

if selected == 'Conclusion':
    st.title('Conclusion')

if selected == 'About Me':
    st.title('YOOOO')


#st.set_page_config(page_title = 'Analysis of Data Scientist Openings', page_icon = '💰', layout='wide')

image = Image.open('image2.png')
st.image(image, width=800)

st.write("""
    We long gone past the time when we once lived where every action required user input and the view on how math subjects can be applicable outside of what is required while in school.  Thanks to every growing field on Artificial Intelligence (AI) and Machine Learning (ML) and the well renowned [article published in Harvard Business Review](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century) back in 2012, which called Data Science (DS) the sexiest job of the 21st century. As businesses continue to seek ways effectively access trends (from present data, reviewing trends from past data and accurately predicting future trends), satisfy the need of customers while operating in the most efficient way possible.

    It is widely known that pursing a career in Data Science can be rewarding in terms of income, while that may be the case as the great saying goes 'More money, more problems'. This purpose of this app will explore through the job listings from a Glassdoor back in 2016 and observe certain trends in salaries, location, satisfaction of a position, the abundance of positions and see how they all relate to one another.""")