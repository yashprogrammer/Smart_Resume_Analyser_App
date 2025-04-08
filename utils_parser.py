import os
import re
import nltk
import spacy
import pandas as pd
import pdfminer.high_level
import pdfminer.layout

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If specific model fails, try using default model
    nlp = spacy.load('en')

# Define skills list
SKILLS_DB = [
    'machine learning', 'data science', 'python', 'word', 'excel', 'English', 'powerpoint', 'sql', 'tableau', 
    'power bi', 'ai', 'artificial intelligence', 'nlp', 'natural language processing', 'tensorflow', 'keras', 
    'pytorch', 'deep learning', 'statistics', 'regression', 'classification', 'clustering', 'react', 'angular', 
    'node', 'javascript', 'typescript', 'vue', 'java', 'c++', 'c#', 'php', 'html', 'css', 'mongodb', 'mysql', 
    'postgresql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git', 'github', 'jira', 'agile', 
    'scrum', 'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking', 'project management',
    'django', 'flask', 'fastapi', 'restful api', 'graphql', 'linux', 'unix', 'windows', 'macos',
    'android', 'ios', 'swift', 'kotlin', 'flutter', 'react native', 'mobile development', 'web development',
    'ui/ux', 'figma', 'sketch', 'adobe xd', 'illustrator', 'photoshop', 'indesign', 'analytics', 
    'seo', 'digital marketing', 'content writing', 'copywriting', 'editing', 'proofreading'
]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        text = pdfminer.high_level.extract_text(file)
    return text

def extract_name(text):
    """Extract name from text using spaCy's NER"""
    doc = nlp(text.strip())
    names = []
    
    # First attempt - look for PERSON entities
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            names.append(ent.text)
    
    if names:
        # Return the first name found - usually the candidate's name appears early in the resume
        return names[0]
    
    # If no name found, return empty string
    return ""

def extract_phone_number(text):
    """Extract phone number using regex"""
    # Regular expression for common phone number formats
    pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?(?:\d{3}[-.\s]?\d{4}))'
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[0]
    return ""

def extract_email(text):
    """Extract email using regex"""
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[0]
    return ""

def extract_skills(text):
    """Extract skills from text by matching with skills database"""
    skills = []
    # Convert text to lowercase for better matching
    text_lower = text.lower()
    
    # Look for each skill in the text
    for skill in SKILLS_DB:
        if skill.lower() in text_lower:
            skills.append(skill)
    
    # Remove duplicates and return
    return list(set(skills))

def count_pages(pdf_path):
    """Count the number of pages in a PDF"""
    page_count = 0
    with open(pdf_path, 'rb') as file:
        # This is a simple way to count pages using pdfminer
        for page in pdfminer.high_level.extract_pages(file):
            page_count += 1
    return page_count

class ResumeParser:
    def __init__(self, resume_path):
        self.resume_path = resume_path
        
    def get_extracted_data(self):
        """Extract all relevant data from resume"""
        if not os.path.exists(self.resume_path):
            return None
        
        # Extract text from the PDF
        text = extract_text_from_pdf(self.resume_path)
        
        if not text:
            return None
        
        # Extract different components
        name = extract_name(text)
        phone = extract_phone_number(text)
        email = extract_email(text)
        skills = extract_skills(text)
        page_count = count_pages(self.resume_path)
        
        # Create a dictionary with all the extracted information
        data = {
            'name': name,
            'mobile_number': phone,
            'email': email,
            'skills': skills,
            'no_of_pages': page_count
        }
        
        return data 