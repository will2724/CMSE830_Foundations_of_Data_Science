import streamlit as st
from PIL import Image
st.set_page_config(page_title = 'Analysis of Data Scientist Openings',
                   page_icon = '💰',
                   layout='wide')


col1, col2 = st.columns([3,2])

with col1:
    image = Image.open('image2.png')
    st.image(image, width=600)
    st.title('Conclusion')
    st.write('''Throughout the exploration of this dataset certain factors have been analyzed to support your journey as you try to find the data science position that is right for you. It was also shown that the more money you make in a position equaites to having a higher level of happiness. This process included reviewal of underlying commonalities in the listings, an evaluation of different visualizations was made to help make this easier, including the comparing and contrasting of the relationship agamous Salary, Employee Satisfaction and their Location. I plan to further investigate this dataset by conducting testing and training of the data, this will allow for better accurate imputation of the missing data.''')

with col2:
    image2 = Image.open('IMG_0370.png')
    st.image(image2, width=600)
    st.write('''A devoted Research Assistant with ~5 years of experience collaborating with a community advancing research by
sequencing primarily in the areas of single-cell, long read DNA, and metagenomics. Committed to using computational
techniques to solve community problems, furthering my professional development through enrolling in courses and
completing projects, while working independently and collaboratively as a data scientist/bioinformatician.''')


