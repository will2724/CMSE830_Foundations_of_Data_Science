df = pd.read_csv('data.csv')
df.pop('Unnamed: 0')
df['age'] = [2023 - i  if i != -1 else i for i in df['Founded']]
df = df.rename(columns = {'Type of ownership' : 'Type of Ownership',
                      'min_salary' : 'Min. Salary',
                      'max_salary' : 'Max. Salary',
                      'avg_salary' : 'Avg. Salary',
                      'job_state' : 'Job State',
                      'same_state' : 'Same State',
                      'age' : 'Age',
                      'python_yn' : 'Python Exp.',
                      'R_yn' : 'R Exp.',
                      'spark' : 'Spark Exp.',
                      'aws' : 'AWS Exp.',
                      'excel' : 'Excel Exp.',
                      'job_simp' : 'Title Simplified',
                     'Headquarters' : 'HQ',
                     'hourly' : 'Hourly',
                     'desc_len' : 'Description Length',
                     'num_comp' : '# of Competitors',
                     'employer_provided' : 'Employer Provided',
                     'seniority' : 'Seniority',
                     })
df['Title Simplified'] = df['Title Simplified'].str.replace('Mle', 'MLE')
df['Min. Salary'] = df['Min. Salary']*1000
df['Max. Salary'] = df['Max. Salary']*1000
df['Avg. Salary'] = df['Avg. Salary']*1000

df_stats = df.drop(['Unnamed: 0', 'Job Title', 'Salary Estimate', 'Job Description', 'Company Name', 'Location', 'HQ', 'Size', 'Type of Ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
       'Hourly', 'Employer Provided', 'company_txt', 'Job State', 'Same State', 'Title Simplified', 'Seniority', '# of Competitors'], axis=1)
df_stats_cols = df_stats.columns
#======================================================================================

tab1, tab2 , tab3 , tab4 ,tab5 = st.tabs(['IDA', 'Scaling','EDA','',''])
@st.cache
with tab1:
    st.markdown('IDA')
    if st.checkbox('Preview Data'):
        st.table(df.head())

    data_shape = st.radio('What is the dimension of:', ('Entire Dataset', 'Rows', 'Columns'))
    if data_shape == 'Entire Dataset':
        st.text('Entire Dataset Shown')
        st.write(df.shape)
    elif data_shape == 'Rows':
        st.text('Rows Shown')
        st.write(df.shape[0])
    else:
        st.text('Columns Shown')
        st.write(df.shape[1])

    if st.checkbox('Check for NaNs'):
        st.write(df.isna().any())


#have tab that can groupby job title and display salary of that title by state
with tab3:
    option_box = st.selectbox(
    st.subheader('What graphs are you inttered in viewing?'),
    ('Histograms', 'Pairplot', 'Map'),
    index=None,
    placeholder='Choose an EDA method'
    )
    if option_box == 'Histograms':
        option_hist = st.selectbox('Choose', df_stats_cols)
    st.write('You selected:', option_hist)
    plot = sns.histplot(df_stats[option_hist])
    st.pyplot(plot.get_figure())

    if option_box == 'Pairplot':
        st.write('You selected to view Pairplot'
        st.pyplot(sns.pairplot(df[Rating', 'Founded', 'Min. Salary', 'Max. Salary', 'Avg. Salary', 'Age', 'Description Length'].get_figure()))

    if option_box == 'Map':
        fig_map = st.radio("State-by-State,  What are you curious to explore?", ('Oppurtunities 🧑‍💻 ', 'Salaries💰', 'Enjoyment 🎭'))
        if fig_map == 'Oppurtunities 👩‍💻 🧑‍💻 👨‍💻':
            fig_states = px.choropleth(height = 800, width = 800,
                locations = df['Job State'].value_counts().index,
                locationmode = 'USA-states',
                color = df['Job State'].value_counts(),
                color_continuous_scale = 'balance',
                labels = {'color': 'Job Openings'},
                title = 'Jobs per State')
            st.pyplot(fig.update_layout(geo_scope = 'usa'))
            st.pyplot(plot.get_figure())
        if fig_map == 'Salaries💰 💳':
            fig_salaries = px.choropleth(height = 800, width = 800,
                locations= df.groupby('Job State')['Avg. Salary'].mean().index,
                locationmode = 'USA-states',
                color = round(df.groupby('Job State')['Avg. Salary'].mean(), 2),
                color_continuous_scale = 'balance',
                labels = {'color':'Yearly Salary'},
                title = 'Average Salary per State')
            plt.update_layout(geo_scope='usa')
            plt.show()
        if fig_map = Enjoyment 🎭':
            fig_rating = px.choropleth(height = 800, width = 800,
                locations = df.groupby('Job State')['Rating'].mean().index,
                locationmode = 'USA-states',
                color = round(df.groupby('Job State')['Rating'].mean(), 2),
                color_continuous_scale = 'balance',
                labels = {'color':'Employee Satisfaction Rating'},
                title = 'Employee Satisfaction Rating per State')
            plt.update_layout(geo_scope = 'usa')
            plt.show()
