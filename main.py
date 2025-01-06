
#importing libraries
from flask import Flask,request,render_template,session,redirect
import numpy as np
import pandas as pd
import pickle
import secrets
import warnings

#warnings
warnings.filterwarnings("ignore", category=UserWarning)


#load datasets
sym_des=pd.read_csv("Dataset/symptoms_df.csv")
precautions=pd.read_csv("Dataset/precautions_df.csv")
health_advice = pd.read_csv("Dataset/workout_df.csv")
description=pd.read_csv("Dataset/description.csv")
medications=pd.read_csv("Dataset/medications.csv")
diets=pd.read_csv("Dataset/diets.csv")

#load model
svc=pickle.load(open("Models/svc.pkl", 'rb'))



#object of flask
app=Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Set Flask to production mode
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
#=================== Custom helper Function ===================
def helper(dis):
    dis = dis.strip().lower()  # Clean the 'dis' variable before matching
    desc = description[description['Disease'].str.strip().str.lower() == dis]['Description']

    if not desc.empty:
        desc = " ".join(desc.tolist())
    else:
        desc = "No description available....."


    pre=precautions[precautions['Disease'].str.strip().str.lower()==dis.strip().lower()][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre=[pre for pre in pre.values]

    med = medications[medications['Disease'].str.strip().str.lower() == dis.strip().lower()]['Medication']
    med = med.iloc[0]
    if isinstance(med, str):
        med = eval(med)

    health=health_advice[health_advice['disease'].str.strip().str.lower()==dis.strip().lower()]['workout']

    diet=diets[diets['Disease'].str.strip().str.lower()==dis.strip().lower()]['Diet']
    diet = diet.iloc[0]
    if isinstance(diet, str):
      diet = eval(diet)


    return desc,pre,med,health,diet

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',1:"AIDS",14: 'Drug Reaction', 34: 'Peptic ulcer diseae', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 31: 'Migraine', 7: 'Cervical spondylosis', 33: 'Paralysis (brain hemorrhage)', 29: 'Jaundice', 30: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 38: 'Typhoid', 41: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 37: 'Tuberculosis', 10: 'Common Cold', 35: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 39:'Urinary tract infection',18: 'Heart attack', 40: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 32: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 36: 'Psoriasis', 27: 'Impetigo',28:'Influenza (Flu)'}


def predict_disease(patient_symptoms):
    # Define specific mappings for certain symptoms or combinations of symptoms
    specific_mappings = {
        'cough': 'Common Cold',
        'headache': 'Migraine',
        'nausea': 'Gastroenteritis',
        'shivering': 'Influenza (Flu)',
        'skin_rash': 'Fungal Infection',
        'joint_pain': 'Osteoarthristis'
    }

    # Define combined symptom logic
    combined_mappings = {
        ('shivering', 'vomiting'): 'Gastroenteritis',
        ('vomiting', 'shivering'): 'Gastroenteritis',
        ('vomiting', 'shivering', 'cough'): 'Influenza (Flu)',
        ('vomiting', 'shivering', 'high_fever'): 'Influenza (Flu)',
        ('shivering', 'vomiting', 'high_fever'): 'Influenza (Flu)',
        ('vomiting', 'shivering', 'cough', 'high_fever'): 'Influenza (Flu)',
        ('vomiting', 'high_fever'): 'Typhoid'

    }

    # Check if the combination of symptoms has a specific disease mapping
    if tuple(patient_symptoms) in combined_mappings:
        return combined_mappings[tuple(patient_symptoms)]

    # Check if a single symptom has a specific disease mapping
    if len(patient_symptoms) == 1 and patient_symptoms[0] in specific_mappings:
        return specific_mappings[patient_symptoms[0]]


    #Create input vector for model prediction if no specific mapping is found
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1

    # Predict disease using the model
    return diseases_list[svc.predict([input_vector])[0]]


#creating routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        symptoms=request.form.get('symptoms')
        user_symptoms = [s.strip().replace(" ", "_") for s in symptoms.split(',')]

        invalid_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
        if invalid_symptoms:
            warning_message = f"Invalid symptoms or no recommendations found. Please consult a doctor."
            return render_template('index.html', message=warning_message)

        predicted_disease = predict_disease(user_symptoms)

        desc, pre, med, health, diet = helper(predicted_disease)

        return render_template('index.html',predicted_disease=predicted_disease,dis_des=desc,dis_pre=pre,dis_med=med,dis_health=health,dis_diet=diet)


#clear all session data
@app.route('/reset')
def reset():
    session.clear()
    return redirect('/')

if __name__=="__main__":
    app.run(debug=True)
