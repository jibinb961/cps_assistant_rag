import cohere

# Paste your Cohere trial API key here
co = cohere.Client('Rg6ZyU14I8qokFl7cSJqLklRimA9Vv0layYwW3MJ')

# The text you want to embed
texts = [
    """# Master of Professional Studies in Digital Media — Online 
The Master of Professional Studies in Digital Media offers a hands-on, collaborative way to explore core creative concepts and innovative ideas in diverse areas of the digital media world. The program provides a solid preparation in the three core digital media knowledge areas: design thinking, technical competency, and creative expression.

## Take a Quick Look
We’re committed to creating an education as unique as your career path. So, whether your goal is a new career or moving up in your field, our innovative programs will get you going your way.
Enrollment : Full-Time, Part-Time
Entry Terms : Fall, Winter
Completion Time : 15-24 months
F1 Visa Eligible : No
Program Type : Online
## Overview : Explore the rapidly changing world of digital media with the STEM-designated Master of Professional Studies in Digital Media. The program enables you to build on your skills and expertise using cutting-edge technologies and tools as you move through your curriculum. Your core courses in information development, usability, and narrative structure provide a baseline for developing, marketing, and managing content-rich experiences. We offer three concentrations online for you to customize your curriculum including social media, digital media management, and interactive design—usability and development. In your final capstone project, you'll turn theory into practice, working with the guidance of a faculty advisor on a personal proposal or with a small, focused team to channel your passion into a project that provides tangible evidence of your abilities.
This is an online program. : ### More Details
#### Unique Features : * Complete an in-depth capstone project experience working under guidance of a faculty advisor.
* Three concentration options provide specialized focus on distinctive areas of digital media. : * Program format is designed to enhance collaboration and networking opportunities for students.


#### Program Objectives
* Leverage your existing professional experience and knowledge in a technology-centric media world.
* Apply human interaction concepts and systems to a wide range of projects.
* Study character and story development across a variety of digital media.
* Develop and test a variety of user experiences for different media and devices.

#### Career Outlook
Companies of all sizes and industries need creative thinkers to help them build and disseminate compelling digital stories for marketing, messaging, and entertainment. Employment projections vary for each concentration, but demand and reward for User Interface professionals and social media experts continues to expand at an above-average pace.
#### Transfer Credit Opportunities for Certifications from Professional Associations
Successful applicants for a master's degree program in the Social Media concentration with their HubSpot Academy Certification:
* Will receive 4 quarter hours of transfer credit – an 8% savings in tuition.
* Will be able to earn a master’s degree by completing 11 courses, rather than 12
* Transfer credit is awarded for the following course: DGM 6285 Interactive Marketing Fundamentals

## Check out your career prospects 
Beginning your journey with us is a big decision. But it's a smart one. Your field is evolving rapidly. That's why we're constantly innovating our programs to anticipate your industry's needs. So, no matter where your field goes, you can lead the way.
4-8% 
* * Job growth for digital media and marketing, faster than average for all jobs* * 
U.S Bureau of Labor Statistics
Northeastern's signature experience-powered learning model has been at the heart of the university for more than a century. It combines world-class academics with professional practice, allowing you to acquire relevant, real-world skills you can immediately put into action in your current workplace.
This makes a Northeastern education a dynamic, transformative experience, giving you countless opportunities to grow as a professional and person.
[Learn About Getting Real World Experience](https:/cps.northeastern.edu/current-students/co-op/)
"""
]

# Get the embeddings
response = co.embed(
    texts=texts,
    model='embed-english-v3.0',
    input_type="search_document"   # for your markdown chunks
)

# Print the embeddings (each is a list of 1024 floats)
for i, embedding in enumerate(response.embeddings):
    print(f"Embedding for text {i+1}:")
    print(embedding)
    print()
