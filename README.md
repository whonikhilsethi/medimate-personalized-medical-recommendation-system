# Medimate: Personalized Medical Recommendation System 🩺💻

Medimate is a full-stack web application built with Flask that leverages advanced AI algorithms to provide fast, accurate medical insights based on your symptoms. By analyzing your inputs, Medimate quickly matches them to a comprehensive database of medical information, offering potential diagnoses in real-time. This enables users to make informed decisions about their health and take timely action. Medimate also provides tailored recommendations, including treatment options, diet suggestions, and lifestyle tips, making healthcare accessible and personalized.

Medimate has been trained using multiple machine learning models, but the disease prediction is performed using the Support Vector Classifier (SVC) due to its better performance and accuracy.

Whether for minor discomfort or more serious concerns, Medimate empowers users to understand and manage their health effectively.

## Table of Contents 📚
1. [Project Overview](#project-overview)
2. [What Makes Medimate Stand Out?](#what-makes-medimate-stand-out)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Installation Instructions](#installation-instructions)
6. [How to Run](#how-to-run)
7. [Routes](#routes)
8. [File Structure](#file-structure)
9. [Acknowledgments](#acknowledgments)
10. [Feedback & Improvements](#feedback--improvements)

## Project Overview 🚀
Medimate is designed to enhance healthcare accessibility by providing personalized disease predictions based on user-reported symptoms. It empowers users by delivering fast and accurate medical insights that help them take appropriate actions for better health management.

This web application uses AI and machine learning to analyze **132 symptoms** and predict **41 potential diseases**. Medimate also offers tailored recommendations, including disease descriptions, medications, diet plans, and health advice.

## What Makes Medimate Stand Out? 🌟
- **Analyze 132 Symptoms**: Medimate uses advanced algorithms to evaluate your reported symptoms, giving you insights into your health.
- **Uncover 41 Potential Diseases**: With quick analysis, Medimate helps you identify possible diseases, giving you the clarity you need to act fast.
- **Receive Tailored Recommendations**: Get personalized advice, including disease descriptions, medications, health tips, and diet recommendations.
- **Precautions and Next Steps**: Based on your symptoms, Medimate provides actionable recommendations on precautions and when to consult a healthcare professional.

## Features ⚙️
- **Symptom-based Disease Prediction**: Users input symptoms, and Medimate predicts the disease using a trained machine learning model.
- **Disease Description**: Detailed descriptions of the diseases predicted by Medimate.
- **Precautions**: Offers a set of precautions to follow for the diagnosed disease.
- **Medications**: Recommends medications tailored to the disease.
- **Health Advice**: Provides personalized health-related advice, including workout suggestions.
- **Diet Recommendations**: Suggests diet plans tailored to the disease.

## Technologies Used 🛠️
- **Flask**: Web framework for building the backend of the application.
- **Python**: Programming language for backend development and AI model integration.
- **Pickle**: Used for loading and saving trained machine learning models.
- **Pandas**: For data manipulation and processing.
- **NumPy**: For numerical computations.
- **Scikit-Learn**: For machine learning algorithms used to predict diseases.
- **HTML/CSS**: For frontend web development.
- **JavaScript**: For frontend interactivity and dynamic content rendering.
- **Secrets**: For secure token generation for user sessions.
- **Warnings**: To manage warning messages in the application.

## Installation Instructions ⚡
1. Clone this repository:
   
   ```bash
   git clone https://github.com/whonikhilsethi/medimate-personalized-medical-recommendation-system.git
   
3. Navigate to the project directory:
   
   ```bash
   cd medimate

5. Install the required Python packages:
   
   ```bash
   pip install -r requirements.txt

## How to Run 🚀
After setting up the environment, follow these steps to run the application locally:
1. Navigate to the project directory (if not already there):

   ```bash
   cd medimate

2. Run the Flask application:
   
   ```bash
   python app.py
3. Open your browser and go to http://127.0.0.1:5000/ to start using Medimate.

## Routes 🌐
* **Home (/):** Displays the homepage where users can input their symptoms to receive disease predictions.
* **Predict (/predict):** Accepts user input for symptoms and returns predicted diseases along with health recommendations.
* **About (/about):** Displays information about the project and its purpose.

## Acknowledgments 🙏
Thanks to the open-source community for the tools and libraries used in this project.
Special thanks to my teammates: [@Darshh Chhabra](https://github.com/darshh009), [@Prason Jena](https://github.com/Prason2912), and [@Bhavish Makkar](https://github.com/Bhavish-Makkar) for their valuable contributions to the project.

## Feedback & Improvements 💬

For feedback or suggestions, feel free to reach out via email at **nikhilsethi2k3@gmail.com**. 📧

This project is continuously evolving, and we welcome your insights to help improve and enhance its features. 🚀💡









