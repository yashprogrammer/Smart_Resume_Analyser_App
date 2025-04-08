import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
spacy.load('en_core_web_sm')

import pandas as pd
import base64, random
import time, datetime 
import sqlite3
import streamlit.components.v1 as components
import os
import pafy
import plotly.express as px
import youtube_dl
import requests
import json
import os
from io import BytesIO
from config import GROK_API_URL, GROK_API_KEY, GROK_MODEL, DB_HOST, DB_USER, DB_PASSWORD, OPENAI_API_KEY
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import io
import pdfminer.high_level
import openai
# Import bokeh components for speech-to-text
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

# Import our custom utils instead of pyresparser
from utils_parser import ResumeParser

def fetch_yt_video(link):
    video = pafy.new(link)
    return video.title


def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file):
    """
    Extract text from PDF file using pdfminer.high_level
    """
    with open(file, 'rb') as f:
        text = pdfminer.high_level.extract_text(f)
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Use more responsive height settings with percentage-based height
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" style="min-height: 800px; height: 80vh;" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


connection = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
cursor = connection.cursor()



def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills,
                courses):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (
    name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills,
    courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


def generate_cover_letter(resume_text, company_name, job_title, job_description):
    """
    Generate a cover letter using the Grok API based on resume and job details
    
    Args:
        resume_text (str): Text extracted from the resume
        company_name (str): Name of the company
        job_title (str): Position title
        job_description (str): Description of the job
    
    Returns:
        str: Generated cover letter or error message
    """
    # Create the prompt for the cover letter generation
    prompt = f"""
    Please generate a professional cover letter based on the following information:
    
    RESUME:
    {resume_text}
    
    JOB DETAILS:
    Company: {company_name}
    Position: {job_title}
    Job Description: {job_description}
    
    Write a compelling cover letter that highlights the relevant skills and experiences from the resume that match the job requirements.
    The cover letter should be professionally formatted with proper sections including date, greeting, introduction, body paragraphs, conclusion, and signature.
    """
    
    # Prepare the request payload
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        # Extract the generated cover letter from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            cover_letter = response_data["choices"][0]["message"]["content"]
            return cover_letter
        else:
            return "Error: Unable to generate cover letter. API response did not contain expected data."
    
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"


def interview_preparation_chat(resume_text, job_role, user_message=None, experience_level=None, specific_skills=None):
    """
    Generates interview preparation responses using the Groq API
    
    Args:
        resume_text (str): Text extracted from the resume
        job_role (str): The job role the user is preparing for
        user_message (str, optional): User's message in the chat. Defaults to None for initial setup.
        experience_level (str, optional): User's experience level. Defaults to None.
        specific_skills (list, optional): List of specific skills for the role. Defaults to None.
    
    Returns:
        str: Response from the Groq API or error message
    """
    # Initial system message to set up the interview preparation scenario
    system_message = f"""
    You are an expert interview coach who helps candidates prepare for job interviews.
    Your task is to simulate a realistic job interview for a {job_role} position.
    
    Based on the candidate's resume, ask relevant technical and behavioral questions.
    Provide honest feedback on answers and suggest improvements.
    
    Keep questions challenging but appropriate for the candidate's experience level.
    Focus on skills and experiences mentioned in their resume.
    """
    
    # Add experience level context if available
    if experience_level:
        system_message += f"""
        
        The candidate's experience level is: {experience_level}
        Please adjust the difficulty and depth of technical questions accordingly.
        """
    
    # Add specific skills context if available
    if specific_skills and len(specific_skills) > 0:
        skills_str = ", ".join(specific_skills)
        system_message += f"""
        
        The candidate has highlighted these specific skills for the role: {skills_str}
        Please include questions that assess proficiency in these skills.
        """
    
    # Add resume information
    system_message += f"""
    
    Resume information:
    {resume_text}
    
    Conduct a professional interview focusing on both technical skills and behavioral aspects.
    Provide constructive feedback after each answer and guide the candidate to improve.
    When the interview is complete, provide a comprehensive assessment.
    """
    
    messages = [{"role": "system", "content": system_message}]
    
    # If no user message (first interaction), provide initial interview setup message
    if user_message is None:
        messages.append({
            "role": "assistant", 
            "content": f"Hello! I'll be conducting a mock interview for the {job_role} position. I'll ask you questions based on your resume and the role requirements. Let's start with a common opening question: Can you tell me about yourself and why you're interested in this {job_role} position?"
        })
    else:
        # Add the user's message and continue the conversation
        messages.append({"role": "user", "content": user_message})
    
    # Prepare the API request
    payload = {
        "model": GROK_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        # Extract the response from the LLM
        if "choices" in response_data and len(response_data["choices"]) > 0:
            interview_response = response_data["choices"][0]["message"]["content"]
            return interview_response
        else:
            return "Error: Unable to generate interview response. API response did not contain expected data."
    
    except Exception as e:
        return f"Error in interview preparation: {str(e)}"


def get_cover_letter_download_link(cover_letter_text, filename="Cover_Letter.txt", link_text="Download Cover Letter"):
    """
    Generates a link to download the cover letter as a text file
    
    Args:
        cover_letter_text (str): The generated cover letter text
        filename (str): The name of the file to download
        link_text (str): The text to display for the download link
    
    Returns:
        str: HTML link to download the cover letter
    """
    # Encode the cover letter text as base64
    b64 = base64.b64encode(cover_letter_text.encode()).decode()
    
    # Create an HTML href link with the base64 data
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    
    return href


def get_interview_transcript_download_link(interview_history, job_role, filename=None, link_text="Download Interview Transcript"):
    """
    Generates a link to download the interview transcript as a text file
    
    Args:
        interview_history (list): List of dictionaries containing the chat history
        job_role (str): The job role the interview was for
        filename (str, optional): The name of the file to download. Defaults to None.
        link_text (str, optional): The text to display for the download link. Defaults to "Download Interview Transcript".
    
    Returns:
        str: HTML link to download the interview transcript
    """
    if filename is None:
        # Generate a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Interview_Transcript_{job_role.replace(' ', '_')}_{timestamp}.txt"
    
    # Format the interview transcript
    transcript_lines = [f"Interview Transcript for {job_role} Position\n", 
                       f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                       "="*50 + "\n\n"]
    
    for message in interview_history:
        if message["role"] == "user":
            transcript_lines.append(f"Candidate: {message['content']}\n\n")
        else:
            transcript_lines.append(f"Interviewer: {message['content']}\n\n")
    
    transcript_text = "".join(transcript_lines)
    
    # Encode the transcript text as base64
    b64 = base64.b64encode(transcript_text.encode()).decode()
    
    # Create an HTML href link with the base64 data
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    
    return href


def generate_project_recommendations(resume_text, resume_data, num_projects=3):
    """
    Generate project recommendations based on resume data
    
    Args:
        resume_text (str): Text extracted from the resume
        resume_data (dict): Parsed resume data
        num_projects (int): Number of projects to recommend
    
    Returns:
        list: List of dictionaries containing project recommendations
    """
    # Create a system prompt for the AI to generate relevant project recommendations
    skills = resume_data.get('skills', [])
    skills_str = ", ".join(skills) if skills else "general programming"
    
    # Clean the resume text to avoid f-string issues
    # Remove any potential problematic characters or formatting
    if resume_text:
        # Simple cleaning: remove any characters that might interfere with f-strings
        resume_text = resume_text.replace('{', '').replace('}', '')
        # Limit the length to avoid token issues
        resume_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
    else:
        resume_text = "No resume text available"
    
    # Create the prompt as a regular string with format() method instead of f-string
    prompt = """
    Based on the following resume information, suggest {num_projects} project ideas that would enhance the resume and showcase the person's skills:
    
    RESUME TEXT:
    {resume_text}
    
    SKILLS:
    {skills_str}
    
    For each project, provide:
    1. Project title
    2. Description (2-3 sentences)
    3. Key technologies/frameworks to use
    4. Skills that will be demonstrated
    5. Difficulty level (Beginner, Intermediate, Advanced)
    6. Estimated time to complete
    7. A source link or tutorial that can help get started (specific URL if possible)
    
    Format your response as structured JSON with the following format for each project:
    
    ```json
    [
      {{
        "title": "Project title",
        "description": "Project description",
        "technologies": ["Tech1", "Tech2", "Tech3"],
        "skills": ["Skill1", "Skill2", "Skill3"],
        "difficulty": "Difficulty level",
        "time_estimate": "Estimated time",
        "source_link": "URL to tutorial or inspiration"
      }}
    ]
    ```
    
    IMPORTANT: 
    - Ensure projects are relevant to the person's existing skills but also introduce new technologies that complement them
    - Projects should be realistic and doable, not overly ambitious
    - Include a mix of difficulty levels
    - Only include legitimate, working links in the source_link field
    - Make sure the JSON is valid and properly formatted
    """.format(
        num_projects=num_projects,
        resume_text=resume_text,
        skills_str=skills_str
    )
    
    # Prepare the request payload
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2500
    }
    
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        response_data = response.json()
        
        # Extract the generated project recommendations from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            project_response = response_data["choices"][0]["message"]["content"]
            
            # Extract JSON from the response
            import re
            import json
            
            # Find JSON content within the response
            json_match = re.search(r'```json\s*(.*?)\s*```', project_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no ```json``` markers, try to find array directly
                json_match = re.search(r'\[\s*\{.*\}\s*\]', project_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to the full response
                    json_str = project_response
            
            try:
                # Parse the JSON
                projects = json.loads(json_str)
                return projects
            except json.JSONDecodeError:
                # If parsing fails, return a formatted error
                return [{"title": "Error parsing project recommendations", 
                         "description": "Unable to parse the AI response. Please try again.",
                         "technologies": [],
                         "skills": [],
                         "difficulty": "N/A",
                         "time_estimate": "N/A",
                         "source_link": ""}]
        else:
            return [{"title": "Error generating project recommendations", 
                     "description": "API response did not contain expected data.",
                     "technologies": [],
                     "skills": [],
                     "difficulty": "N/A",
                     "time_estimate": "N/A",
                     "source_link": ""}]
    
    except Exception as e:
        return [{"title": "Error in project recommendations", 
                 "description": f"Error: {str(e)}",
                 "technologies": [],
                 "skills": [],
                 "difficulty": "N/A",
                 "time_estimate": "N/A",
                 "source_link": ""}]


def get_project_recommendations_download_link(projects, filename=None, link_text="Download Project Recommendations"):
    """
    Generates a link to download the project recommendations as a text file
    
    Args:
        projects (list): List of project recommendation dictionaries
        filename (str, optional): The name of the file to download. Defaults to None.
        link_text (str, optional): The text to display for the download link.
    
    Returns:
        str: HTML link to download the project recommendations
    """
    if filename is None:
        # Generate a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Project_Recommendations_{timestamp}.txt"
    
    # Format the project recommendations
    lines = ["# PROJECT RECOMMENDATIONS\n", 
             f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             "="*50 + "\n\n"]
    
    for i, project in enumerate(projects, 1):
        lines.append(f"## PROJECT {i}: {project.get('title', 'Unknown')}\n\n")
        lines.append(f"Description: {project.get('description', 'No description')}\n\n")
        
        technologies = project.get('technologies', [])
        if technologies:
            lines.append("Technologies:\n")
            for tech in technologies:
                lines.append(f"- {tech}\n")
            lines.append("\n")
        
        skills = project.get('skills', [])
        if skills:
            lines.append("Skills demonstrated:\n")
            for skill in skills:
                lines.append(f"- {skill}\n")
            lines.append("\n")
        
        lines.append(f"Difficulty: {project.get('difficulty', 'Unknown')}\n")
        lines.append(f"Estimated time: {project.get('time_estimate', 'Unknown')}\n")
        lines.append(f"Source/Tutorial: {project.get('source_link', 'Not provided')}\n\n")
        lines.append("-"*50 + "\n\n")
    
    content = "".join(lines)
    
    # Encode the text as base64
    b64 = base64.b64encode(content.encode()).decode()
    
    # Create an HTML href link with the base64 data
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    
    return href


st.set_page_config(
    page_title="Career Boost",
    page_icon='./Logo/SRA_Logo.ico',
    layout="wide",  # Use wide layout to maximize space
)

# Add custom CSS for fixing the height issue


def run():
    # Main title now set dynamically based on selected feature
    # Default title for when no feature is selected
    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = None
    
    # Define audio recording handler route for Streamlit Component
    if st.experimental_get_query_params().get("__audio_recorded", False):
        try:
            audio_data = st.experimental_get_query_params()["__audio_data"][0]
            audio_bytes = base64.b64decode(audio_data)
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_recorded = True
            st.session_state.recording = False
            # Clear the query params
            params = st.experimental_get_query_params()
            if "__audio_recorded" in params:
                del params["__audio_recorded"]
            if "__audio_data" in params:
                del params["__audio_data"]
            st.experimental_set_query_params(**params)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
    
    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS SRA;"""
    cursor.execute(db_sql)
    connection.select_db("sra")

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                     Name varchar(100) NOT NULL,
                     Email_ID VARCHAR(50) NOT NULL,
                     resume_score VARCHAR(8) NOT NULL,
                     Timestamp VARCHAR(50) NOT NULL,
                     Page_no VARCHAR(5) NOT NULL,
                     Predicted_Field VARCHAR(25) NOT NULL,
                     User_level VARCHAR(30) NOT NULL,
                     Actual_skills VARCHAR(300) NOT NULL,
                     Recommended_skills VARCHAR(300) NOT NULL,
                     Recommended_courses VARCHAR(600) NOT NULL,
                     PRIMARY KEY (ID));
                    """
    cursor.execute(table_sql)
    
    if True:
        # Initialize session state for resume if it doesn't exist
        if 'resume_uploaded' not in st.session_state:
            st.session_state.resume_uploaded = False
            st.session_state.resume_path = None
            st.session_state.resume_text = None
            st.session_state.resume_data = None
        
        # Initialize interview-related session state variables
        if 'interview_start_time' not in st.session_state:
            st.session_state.interview_start_time = None
        
        if 'interview_history' not in st.session_state:
            st.session_state.interview_history = []
            st.session_state.interview_started = False
            st.session_state.experience_level = None
            st.session_state.specific_skills = []
        
        # Ensure user_input is always initialized
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        
        # Resume upload in sidebar
        st.sidebar.markdown("# Upload Resume")
        pdf_file = st.sidebar.file_uploader("Upload your Resume", type=["pdf"], 
                                     help="Upload your resume in PDF format. This will be used for all features.")
        
        if pdf_file is not None:
            # Save the uploaded file
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            
            # Check if this is a new upload or already processed
            if not st.session_state.resume_uploaded or st.session_state.resume_path != save_image_path:
                with open(save_image_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                # Process the resume
                with st.spinner('Processing your resume...'):
                    # Extract resume data
                    resume_data = ResumeParser(save_image_path).get_extracted_data()
                    
                    # Extract text from the resume
                    resume_text = pdf_reader(save_image_path)
                    
                    # Store in session state
                    st.session_state.resume_uploaded = True
                    st.session_state.resume_path = save_image_path
                    st.session_state.resume_text = resume_text
                    st.session_state.resume_data = resume_data
                
                st.sidebar.success(f"Resume uploaded: {pdf_file.name}")
                
                # Display basic info in sidebar
                if resume_data and 'name' in resume_data:
                    st.sidebar.markdown(f"**Hello {resume_data['name']}!**")
        
        # Display features only after resume is uploaded
        if st.session_state.resume_uploaded:
            # Add a feature selection dropdown
            st.sidebar.markdown("# Select Feature")
            features = ["Resume Analysis", "Generate Cover Letter", "Interview Preparation", "Project Recommender"]
            selected_feature = st.sidebar.selectbox("Choose a feature:", features)
            
            # Store selected feature in session state
            st.session_state.selected_feature = selected_feature
            
            # Set the page title based on the selected feature
            if selected_feature == "Resume Analysis":
                st.title("Analyse your Resume")
                
                # Display content in a single column layout
                # Show the uploaded PDF
                st.header("Your Resume")
                show_pdf(st.session_state.resume_path)
                
                # Get resume data and text from session state
                resume_data = st.session_state.resume_data
                resume_text = st.session_state.resume_text
                
                if resume_data:
                    st.header("**Resume Analysis**")
                    st.success("Hello " + resume_data['name'])
                    st.subheader("**Your Basic info**")
                    try:
                        st.text('Name: ' + resume_data['name'])
                        st.text('Email: ' + resume_data['email'])
                        st.text('Contact: ' + resume_data['mobile_number'])
                        st.text('Resume pages: ' + str(resume_data['no_of_pages']))
                    except:
                        pass
                    
                    # Rest of the resume analysis code remains the same
                    cand_level = ''
                    if resume_data['no_of_pages'] == 1:
                        cand_level = "Fresher"
                        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',
                                    unsafe_allow_html=True)
                    elif resume_data['no_of_pages'] == 2:
                        cand_level = "Intermediate"
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',
                                    unsafe_allow_html=True)
                    elif resume_data['no_of_pages'] >= 3:
                        cand_level = "Experienced"
                        st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',
                                    unsafe_allow_html=True)

                    st.subheader("**Skills Recommendation💡**")
                    ## Skill shows
                    keywords = st_tags(label='### Skills that you have',
                                      text='See our skills recommendation',
                                      value=resume_data['skills'], key='1')

                    ##  recommendation
                    ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask',
                                'streamlit']
                    web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                                  'javascript', 'angular js', 'c#', 'flask']
                    android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
                    ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
                    uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
                                    'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator',
                                    'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro',
                                    'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp',
                                    'user research', 'user experience']

                    recommended_skills = []
                    reco_field = ''
                    rec_course = ''
                    ## Courses recommendation
                    for i in resume_data['skills']:
                        ## Data science recommendation
                        if i.lower() in ds_keyword:
                            print(i.lower())
                            reco_field = 'Data Science'
                            st.success("** Our analysis says you are looking for Data Science Jobs.**")
                            recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                                  'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                                  'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                                  'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                                  'Streamlit']
                            recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                           text='Recommended skills generated from System',
                                                           value=recommended_skills, key='2')
                            st.markdown(
                                '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                                unsafe_allow_html=True)
                            rec_course = course_recommender(ds_course)
                            break

                        ## Web development recommendation
                        elif i.lower() in web_keyword:
                            print(i.lower())
                            reco_field = 'Web Development'
                            st.success("** Our analysis says you are looking for Web Development Jobs **")
                            recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                                  'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                            recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                           text='Recommended skills generated from System',
                                                           value=recommended_skills, key='3')
                            st.markdown(
                                '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                                unsafe_allow_html=True)
                            rec_course = course_recommender(web_course)
                            break

                        ## Android App Development
                        elif i.lower() in android_keyword:
                            print(i.lower())
                            reco_field = 'Android Development'
                            st.success("** Our analysis says you are looking for Android App Development Jobs **")
                            recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                                  'Kivy', 'GIT', 'SDK', 'SQLite']
                            recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                           text='Recommended skills generated from System',
                                                           value=recommended_skills, key='4')
                            st.markdown(
                                '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                                unsafe_allow_html=True)
                            rec_course = course_recommender(android_course)
                            break

                        ## IOS App Development
                        elif i.lower() in ios_keyword:
                            print(i.lower())
                            reco_field = 'IOS Development'
                            st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                            recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                                  'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                                  'Auto-Layout']
                            recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                           text='Recommended skills generated from System',
                                                           value=recommended_skills, key='5')
                            st.markdown(
                                '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                                unsafe_allow_html=True)
                            rec_course = course_recommender(ios_course)
                            break

                        ## Ui-UX Recommendation
                        elif i.lower() in uiux_keyword:
                            print(i.lower())
                            reco_field = 'UI-UX Development'
                            st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                            recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                                                  'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                                                  'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                                                  'Solid', 'Grasp', 'User Research']
                            recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                           text='Recommended skills generated from System',
                                                           value=recommended_skills, key='6')
                            st.markdown(
                                '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                                unsafe_allow_html=True)
                            rec_course = course_recommender(uiux_course)
                            break

                    #
                    ## Insert into table
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date + '_' + cur_time)

                    ### Resume writing recommendation
                    st.subheader("**Resume Tips & Ideas💡**")
                    resume_score = 0
                    if 'Objective' in resume_text:
                        resume_score = resume_score + 20
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career objective, it will give your career intension to the Recruiters.</h4>''',
                            unsafe_allow_html=True)

                    if 'Declaration' in resume_text:
                        resume_score = resume_score + 20
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration✍</h4>''',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',
                            unsafe_allow_html=True)

                    if 'Hobbies' or 'Interests' in resume_text:
                        resume_score = resume_score + 20
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',
                            unsafe_allow_html=True)

                    if 'Achievements' in resume_text:
                        resume_score = resume_score + 20
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅 </h4>''',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Achievements🏅. It will show that you are capable for the required position.</h4>''',
                            unsafe_allow_html=True)

                    if 'Projects' in resume_text:
                        resume_score = resume_score + 20
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h4>''',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h4>''',
                            unsafe_allow_html=True)

                    st.subheader("**Resume Score📝**")
                    st.markdown(
                        """
                        <style>
                            .stProgress > div > div > div > div {
                                background-color: #d73b5c;
                            }
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    my_bar = st.progress(0)
                    score = 0
                    for percent_complete in range(resume_score):
                        score += 1
                        time.sleep(0.1)
                        my_bar.progress(percent_complete + 1)
                    st.success('** Your Resume Writing Score: ' + str(score) + '**')
                    st.warning(
                        "** Note: This score is calculated based on the content that you have added in your Resume. **")
                    st.balloons()

                    insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                                str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                                str(recommended_skills), str(rec_course))

                    ## Resume writing video
                    # st.header("**Bonus Video for Resume Writing Tips💡**")
                    # resume_vid = random.choice(resume_videos)
                    # res_vid_title = fetch_yt_video(resume_vid)
                    # st.subheader("✅ **" + res_vid_title + "**")
                    # st.video(resume_vid)

                    ## Interview Preparation Video
                    # st.header("**Bonus Video for Interview👨‍💼 Tips💡**")
                    # interview_vid = random.choice(interview_videos)
                    # int_vid_title = fetch_yt_video(interview_vid)
                    # st.subheader("✅ **" + int_vid_title + "**")
                    # st.video(interview_vid)

                    connection.commit()
                else:
                    st.error('Something went wrong..')
            
            elif selected_feature == "Generate Cover Letter":
                st.title("Generate Cover Letter")
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Provide job details to generate a personalized cover letter</h4>''',
                          unsafe_allow_html=True)
                
                # Get resume text from session state
                resume_text = st.session_state.resume_text
                
                # Get job details from user
                st.subheader("Job Details")
                company_name = st.text_input("Company Name", help="The name of the company you're applying to")
                job_title = st.text_input("Job Title/Position", help="The position you're applying for")
                job_description = st.text_area("Job Description", 
                                             help="Copy and paste the job description here. The more details you provide, the better the cover letter will be tailored to the job.")
                
                # Single button with conditional logic inside the button click handler
                generate_button = st.button("Generate Cover Letter")
                if generate_button:
                    # Check if all required fields are filled
                    if company_name and job_title and job_description:
                        with st.spinner('Generating your personalized cover letter...'):
                            # Call the function to generate the cover letter
                            cover_letter = generate_cover_letter(resume_text, company_name, job_title, job_description)
                            
                            # Display the generated cover letter
                            st.subheader("Your Generated Cover Letter")
                            st.text_area("Cover Letter", cover_letter, height=400)
                            
                            # Provide a download link
                            st.markdown(get_cover_letter_download_link(cover_letter, f"Cover_Letter_{company_name}_{job_title}.txt", "📥 Download Cover Letter"), 
                                      unsafe_allow_html=True)
                            
                            st.success("Cover letter generated successfully! You can download it using the link above.")
                            st.balloons()
                    else:
                        # Handle missing information
                        if not company_name:
                            st.error("Please enter the company name")
                        if not job_title:
                            st.error("Please enter the job title/position")
                        if not job_description:
                            st.error("Please enter the job description")
            
            elif selected_feature == "Interview Preparation":
                st.title("Interview Preparation")
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Provide job details to start interview preparation</h4>''',
                          unsafe_allow_html=True)
                
                # Wrap in a div with ID for CSS targeting
                st.markdown('<div id="interview-preparation">', unsafe_allow_html=True)
                
                # Get resume text from session state
                resume_text = st.session_state.resume_text
                
                # Get job role from user
                st.subheader("Job Details")
                job_role = st.text_input("Job Role", help="The position you're preparing to interview for")
                
                # Additional information (optional)
                experience_level = st.selectbox(
                    "Experience Level", 
                    ["Entry Level", "Mid Level", "Senior Level"],
                    help="Your experience level for the role"
                )
                
                specific_skills = st_tags(
                    label='Key Skills for the Role',
                    text='Enter skills and press enter',
                    value=[],
                    key='interview_skills'
                )
                
                # Column layout for buttons
                col1, col2 = st.columns(2)
                
                # Start interview button
                with col1:
                    start_button = st.button("Start Interview Preparation")
                
                # Reset interview button
                with col2:
                    reset_button = st.button("Reset Interview")
                
                # Handle reset button
                if reset_button:
                    st.session_state.interview_history = []
                    st.session_state.interview_started = False
                    st.session_state.experience_level = None
                    st.session_state.specific_skills = []
                    st.session_state.user_input = ""
                    st.session_state.interview_start_time = None
                    st.success("Interview has been reset. Click 'Start Interview Preparation' to begin a new session.")
                    st.experimental_rerun()
                
                # Start interview logic
                if start_button and job_role:
                    st.session_state.interview_started = True
                    
                    # Initialize the timer at the start of the interview
                    if not st.session_state.interview_start_time:
                        st.session_state.interview_start_time = time.time()
                    
                    # Store the experience level and specific skills in session state
                    st.session_state.experience_level = experience_level
                    st.session_state.specific_skills = specific_skills
                    
                    # Only generate initial response if the history is empty
                    if len(st.session_state.interview_history) == 0:
                        with st.spinner("Preparing interview questions..."):
                            initial_response = interview_preparation_chat(
                                resume_text, 
                                job_role, 
                                experience_level=st.session_state.experience_level, 
                                specific_skills=st.session_state.specific_skills
                            )
                            st.session_state.interview_history.append({"role": "assistant", "content": initial_response})
                
                # Display chat interface if interview is started
                if st.session_state.interview_started:
                    st.subheader("Interview Simulation")
                    
                    # Timer logic
                    if st.session_state.interview_start_time is not None:
                        elapsed_time = time.time() - st.session_state.interview_start_time
                        remaining_time = max(0, 15 * 60 - elapsed_time)  # 15 minutes in seconds
                        
                        # Format remaining time as mm:ss
                        mins = int(remaining_time // 60)
                        secs = int(remaining_time % 60)
                        time_str = f"{mins:02d}:{secs:02d}"
                        
                        # Create timer display
                        timer_container = st.container()
                        with timer_container:
                            # Use columns to center the timer
                            t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
                            with t_col2:
                                # Display timer with conditional styling based on remaining time
                                if remaining_time > 5 * 60:  # More than 5 minutes
                                    timer_color = "#1ed760"  # Green
                                elif remaining_time > 2 * 60:  # Between 2-5 minutes
                                    timer_color = "#FFA500"  # Orange
                                else:  # Less than 2 minutes
                                    timer_color = "#FF0000"  # Red
                                
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <p style="margin-bottom: 5px; font-size: 0.9rem;">Remaining Time:</p>
                                        <div style="
                                            background-color: rgba(0,0,0,0.1); 
                                            border-radius: 10px; 
                                            padding: 10px; 
                                            font-size: 1.5rem; 
                                            font-weight: bold; 
                                            color: {timer_color};">
                                            {time_str}
                                        </div>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                        
                        # Auto-end interview if time is up
                        if remaining_time <= 0 and not st.session_state.get('show_feedback', False):
                            st.warning("⏰ Time's up! The interview session has ended.")
                            st.session_state.show_feedback = True
                            # Add an automatic message to the interview history
                            st.session_state.interview_history.append({
                                "role": "assistant", 
                                "content": "Our 15-minute interview session has concluded due to the time limit. Let me provide you with feedback on your performance so far."
                            })
                            # Force a rerun to show the feedback
                            st.experimental_rerun()
                        
                        # Create a mechanism for auto-refresh
                        # Use a hidden placeholder to trigger reloads
                        placeholder = st.empty()
                        
                        # Use custom HTML and JavaScript for a truly live countdown timer
                        countdown_js = f"""
                        <div id="countdown-container" style="display: none;"></div>
                        <script>
                            // Function to update countdown and reload when needed
                            function updateCountdown() {{
                                // Get current time
                                const now = new Date().getTime();
                                // Get interview start time (convert from seconds to milliseconds)
                                const startTime = {st.session_state.interview_start_time * 1000};
                                // Calculate time remaining
                                const duration = 15 * 60 * 1000; // 15 minutes in milliseconds
                                const elapsed = now - startTime;
                                const remaining = Math.max(0, duration - elapsed);
                                
                                if (remaining <= 0) {{
                                    // Time's up - reload page to trigger the end of interview
                                    window.location.reload();
                                    return;
                                }}
                                
                                // If we still have time, check if Streamlit is connected before scheduling next update
                                if (window.Streamlit) {{
                                    // Schedule next update
                                    setTimeout(updateCountdown, 1000);
                                    
                                    // Force a rerun every 5 seconds to update the UI
                                    if (Math.floor(elapsed / 1000) % 5 === 0) {{
                                        window.location.reload();
                                    }}
                                }}
                            }}
                            
                            // Start the countdown as soon as the script loads
                            updateCountdown();
                        </script>
                        """
                        st.components.v1.html(countdown_js, height=0)
                            
                    # Display chat history in a container with scrolling
                    chat_container = st.container()
                    with chat_container:
                        for message in st.session_state.interview_history:
                            if message["role"] == "user":
                                st.markdown(f"<div style='background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color: #333333; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>Interviewer:</b> {message['content']}</div>", unsafe_allow_html=True)
                    
                    # Create a form to prevent rerun on every input
                    with st.form(key="interview_form"):
                        # Safely access user_input with a default empty string
                        current_input = st.session_state.get('user_input', "")
                        user_input = st.text_area("Your response:", height=100, value=current_input, key="interview_input")
                        
                        # Add a note about paste prevention
                        st.caption("Note: Pasting is disabled to simulate a real interview environment. Use the microphone button for voice typing.")

                        # Create two columns for the Send button and Mic button
                        col1, col2 = st.columns([5, 1])

                        with col1:
                            submit_button = st.form_submit_button("Send")

                        # The Mic button and its logic go in the second column
                        with col2:
                            # Create a Bokeh button for speech-to-text
                            stt_button = Button(label="🎤", width=50, height=38,
                                               button_type="success", css_classes=["mic-button"])

                            # Add JavaScript code for speech recognition
                            stt_button.js_on_event("button_click", CustomJS(code="""
                                // Find the target textarea
                                const textareas = document.querySelectorAll('textarea');
                                let targetTextarea = null;
                                textareas.forEach(function(textarea) {
                                    if (textarea.closest('.stForm')) { // Ensure it's the textarea inside the form
                                        targetTextarea = textarea;
                                    }
                                });

                                if (!targetTextarea) {
                                    console.error("Target textarea not found");
                                    return;
                                }

                                var recognition = new webkitSpeechRecognition();
                                recognition.continuous = true;
                                recognition.interimResults = true;
                                recognition.lang = 'en-US';

                                const originalValue = targetTextarea.value;
                                const recordingIndicator = " 🎤 Recording...";
                                targetTextarea.value = originalValue + recordingIndicator;

                                // Dispatch input event to update Streamlit's view of the textarea value
                                const inputEvent = new Event('input', { bubbles: true });
                                targetTextarea.dispatchEvent(inputEvent);

                                recognition.onresult = function (e) {
                                    var interimTranscript = '';
                                    var finalTranscript = '';
                                    for (var i = e.resultIndex; i < e.results.length; ++i) {
                                        if (e.results[i].isFinal) {
                                            finalTranscript += e.results[i][0].transcript;
                                        } else {
                                            interimTranscript += e.results[i][0].transcript;
                                        }
                                    }
                                    targetTextarea.value = originalValue + finalTranscript + (interimTranscript ? " [" + interimTranscript + "]" : "");
                                    targetTextarea.dispatchEvent(inputEvent); // Update Streamlit continuously
                                }

                                recognition.onerror = function(event) {
                                    console.error('Speech recognition error:', event.error);
                                    targetTextarea.value = originalValue + " ❌ Error: " + event.error;
                                    targetTextarea.dispatchEvent(inputEvent);
                                    recognition.stop();
                                }

                                recognition.onend = function() {
                                    // Clean up the text area - remove indicator and interim brackets
                                    targetTextarea.value = targetTextarea.value.replace(recordingIndicator, "").replace(/\s*\[.*?\]\s*$/, "");
                                    targetTextarea.dispatchEvent(inputEvent); // Final update for Streamlit
                                }

                                recognition.start();

                                // Stop after 15 seconds
                                setTimeout(function() {
                                    if (recognition) {
                                        try { recognition.stop(); } catch(e) { console.error('Error stopping recognition:', e); }
                                    }
                                }, 15000);
                                """))

                            # Use streamlit_bokeh_events to render the button and capture events
                            # Ensure override_height is sufficient for the button
                            voice_result = streamlit_bokeh_events(
                                stt_button,
                                events="GET_TEXT", # We might not need this event if JS updates textarea directly
                                key="listen",
                                refresh_on_update=False,
                                override_height=50, # Ensure space for the button
                                debounce_time=0
                            )

                        # Add CSS specifically for this section
                        st.markdown("""
                        <style>
                        /* Style the form to align items better */
                        div.stForm > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                            gap: 0.5rem; /* Adjust spacing between elements */
                        }
                        /* Ensure columns are vertically aligned */
                        div.stForm div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                            display: flex;
                            flex-direction: column;
                            justify-content: flex-end; /* Align button bottom */
                        }
                         /* Target the specific column containing the mic button for alignment */
                        div.stForm div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
                             padding-top: 28px; /* Adjust this value to align mic button with Send */
                             height: 100%;
                        }
                        /* Style the bokeh button */
                        .mic-button .bk-btn.bk-btn-success {
                            background-color: #1ed760 !important;
                            border-color: #1ed760 !important;
                            color: white !important;
                            font-size: 1.1rem !important;
                            line-height: 1.5 !important;
                            padding: 0.375rem 0.75rem !important;
                        }
                        /* Hide the iframe generated by streamlit_bokeh_events */
                        iframe {
                            display: none !important;
                            height: 0 !important;
                            width: 0 !important;
                            visibility: hidden !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        # Handle form submission (AFTER button definitions)
                        if submit_button and user_input:
                            # Add user message to history
                            st.session_state.interview_history.append({"role": "user", "content": user_input})
                            
                            # Clear the input by setting session state to empty
                            st.session_state.user_input = ""
                            
                            # Get response from LLM with spinner to show processing
                            with st.spinner("Interviewer is thinking..."):
                                bot_response = interview_preparation_chat(
                                    resume_text, 
                                    job_role, 
                                    user_input, 
                                    st.session_state.experience_level, 
                                    st.session_state.specific_skills
                                )
                                st.session_state.interview_history.append({"role": "assistant", "content": bot_response})
                            
                            # Force a rerun to update the chat
                            st.experimental_rerun()
                    
                    # Add JavaScript to disable paste in the response textarea
                    st.markdown("""
                    <script>
                    // Function to run once DOM is loaded
                    document.addEventListener('DOMContentLoaded', function() {
                        // Find all textareas and disable paste
                        setTimeout(function() {
                            const textareas = document.querySelectorAll('textarea');
                            textareas.forEach(function(textarea) {
                                textarea.addEventListener('paste', function(e) {
                                    e.preventDefault();
                                    alert('Pasting is not allowed in the interview simulation.');
                                    return false;
                                });
                            });
                        }, 1000);
                    });
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # End interview and get feedback
                    end_col1, end_col2 = st.columns(2)
                    
                    with end_col1:
                        if st.button("End Interview & Get Feedback"):
                            st.session_state.show_feedback = True
                    
                    with end_col2:
                        # Add download transcript button
                        if len(st.session_state.interview_history) > 0:
                            st.markdown(
                                get_interview_transcript_download_link(
                                    st.session_state.interview_history, 
                                    job_role
                                ), 
                                unsafe_allow_html=True
                            )
                    
                    # Show feedback if requested
                    if st.session_state.get('show_feedback', False):
                        st.subheader("Interview Feedback")
                        with st.spinner("Generating interview feedback..."):
                            # Create a feedback prompt
                            feedback_prompt = f"""
                            Based on our interview session for the {job_role} position, please provide:
                            1. A summary of strengths demonstrated in my answers
                            2. Areas for improvement
                            3. Practical tips for the actual interview
                            4. Sample answers to some key questions that were asked
                            """
                            
                            # Get feedback from the LLM
                            feedback_response = interview_preparation_chat(
                                resume_text, 
                                job_role, 
                                feedback_prompt, 
                                st.session_state.experience_level, 
                                st.session_state.specific_skills
                            )
                            
                            # Display the feedback in a nice format
                            st.markdown(f"<div style='background-color: #e8f5e9; padding: 15px; border-radius: 5px;'>{feedback_response}</div>", unsafe_allow_html=True)
                            
                            # Provide a download link for the feedback
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"Interview_Feedback_{job_role.replace(' ', '_')}_{timestamp}.txt"
                            st.markdown(
                                get_cover_letter_download_link(
                                    feedback_response, 
                                    filename=filename, 
                                    link_text="📥 Download Feedback"
                                ),
                                unsafe_allow_html=True
                            )
                
                elif not job_role and start_button:
                    st.error("Please enter the job role you're preparing for")
                
                # Close the div we opened for CSS targeting
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif selected_feature == "Project Recommender":
                st.title("Project Recommender")
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Discover projects to enhance your resume and showcase your skills</h4>''',
                          unsafe_allow_html=True)
                
                # Get resume data and text from session state
                resume_data = st.session_state.resume_data
                resume_text = st.session_state.resume_text
                
                if resume_data:
                    # Display user's skills
                    st.subheader("Your Skills")
                    if 'skills' in resume_data and resume_data['skills']:
                        skills_str = ", ".join(resume_data['skills'])
                        st.write(skills_str)
                    else:
                        st.info("No skills detected in your resume. Consider adding skills to get better project recommendations.")
                    
                    # Project count selector
                    st.subheader("Project Recommendations")
                    num_projects = st.slider("How many projects would you like to be recommended?", min_value=1, max_value=5, value=3)
                    
                    # Generate recommendations button
                    if st.button("Get Project Recommendations"):
                        with st.spinner("Generating project recommendations based on your resume..."):
                            # Call function to generate project recommendations
                            projects = generate_project_recommendations(resume_text, resume_data, num_projects)
                            
                            # Store projects in session state
                            st.session_state.recommended_projects = projects
                            
                            # Display projects
                            for i, project in enumerate(projects, 1):
                                with st.expander(f"Project {i}: {project.get('title', 'Unknown')}", expanded=True):
                                    st.markdown(f"**Description:** {project.get('description', 'No description')}")
                                    
                                    # Technologies
                                    st.markdown("**Technologies:**")
                                    tech_cols = st.columns(3)
                                    technologies = project.get('technologies', [])
                                    for j, tech in enumerate(technologies):
                                        tech_cols[j % 3].markdown(f"- {tech}")
                                    
                                    # Skills demonstrated
                                    st.markdown("**Skills demonstrated:**")
                                    skill_cols = st.columns(3)
                                    skills = project.get('skills', [])
                                    for j, skill in enumerate(skills):
                                        skill_cols[j % 3].markdown(f"- {skill}")
                                    
                                    # Project details
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**Difficulty:** {project.get('difficulty', 'Unknown')}")
                                    with col2:
                                        st.markdown(f"**Estimated time:** {project.get('time_estimate', 'Unknown')}")
                                    
                                    # Source link
                                    source_link = project.get('source_link', '')
                                    if source_link:
                                        st.markdown(f"**Source/Tutorial:** [Link]({source_link})")
                            
                            # Add download button for the recommendations
                            st.markdown(get_project_recommendations_download_link(projects, link_text="📥 Download Project Recommendations"), 
                                      unsafe_allow_html=True)
                            
                            st.success(f"Successfully generated {num_projects} project recommendations based on your skills!")
                            st.balloons()
                    
                    # Display recommendations from session state if they exist
                    elif 'recommended_projects' in st.session_state:
                        projects = st.session_state.recommended_projects
                        
                        # Display projects
                        for i, project in enumerate(projects, 1):
                            with st.expander(f"Project {i}: {project.get('title', 'Unknown')}", expanded=True):
                                st.markdown(f"**Description:** {project.get('description', 'No description')}")
                                
                                # Technologies
                                st.markdown("**Technologies:**")
                                tech_cols = st.columns(3)
                                technologies = project.get('technologies', [])
                                for j, tech in enumerate(technologies):
                                    tech_cols[j % 3].markdown(f"- {tech}")
                                
                                # Skills demonstrated
                                st.markdown("**Skills demonstrated:**")
                                skill_cols = st.columns(3)
                                skills = project.get('skills', [])
                                for j, skill in enumerate(skills):
                                    skill_cols[j % 3].markdown(f"- {skill}")
                                
                                # Project details
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Difficulty:** {project.get('difficulty', 'Unknown')}")
                                with col2:
                                    st.markdown(f"**Estimated time:** {project.get('time_estimate', 'Unknown')}")
                                
                                # Source link
                                source_link = project.get('source_link', '')
                                if source_link:
                                    st.markdown(f"**Source/Tutorial:** [Link]({source_link})")
                        
                        # Add download button for the recommendations
                        st.markdown(get_project_recommendations_download_link(projects, link_text="📥 Download Project Recommendations"), 
                                  unsafe_allow_html=True)
                else:
                    st.error("Unable to access resume data. Please try uploading your resume again.")
        
        else:
            # Default title when no feature is selected yet
            st.title("Resume Assistant")
            
            # Show prompt to upload resume first
            st.info("📄 Please upload your resume in the sidebar to access features.")
            
            # Optional: Show a preview of available features
            st.markdown("### Available features after resume upload:")
            st.markdown("- **Resume Analysis**: Get insights about your resume and recommendations for improvement")
            st.markdown("- **Cover Letter Generator**: Create customized cover letters for job applications")
            st.markdown("- **Interview Preparation**: Practice with an AI-powered interview simulator")
  

def transcribe_audio_with_whisper(audio_data):
    """
    Transcribe audio using OpenAI's Whisper API
    
    Args:
        audio_data (bytes): Raw audio data to transcribe
    
    Returns:
        str: Transcribed text or error message
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Create a temporary file to store the audio
        temp_audio_file = "temp_recording.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio_data)
        
        # Send the audio file to the Whisper API
        with open(temp_audio_file, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Remove the temporary file
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        
        # Return the transcribed text
        return response.text
    
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

run()
