from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, PasswordField
from wtforms.validators import InputRequired, ValidationError, Email, Length
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config["SECRET_KEY"] = 'secretkey'
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), 'static/files')
app.config["CSV_FILE_PATH"] = os.path.join(os.getcwd(), 'combined_marks_and_topics.csv')
app.config["CHART_FOLDER"] = os.path.join(os.getcwd(), 'static/charts')

# Dummy user data storage
users = {}

class UploadFileForm(FlaskForm):
    file = FileField("file", validators=[InputRequired()])

    def validate_file(form, field):
        if not field.data.filename.lower().endswith('.pdf'):
            raise ValidationError('File must be a PDF')

    submit = SubmitField("Upload File")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[InputRequired(), Email()])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6, max=12)])
    submit = SubmitField("Login")

class SignupForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired(), Length(min=3, max=20)])
    email = StringField("Email", validators=[InputRequired(), Email()])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6, max=12)])
    submit = SubmitField("Sign Up")

# Load and prepare the model
def load_model():
    df = pd.read_csv('history_questions.csv')
    X = df['Question']
    y = df['Topic']
    
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)
    
    return model, vectorizer

model, vectorizer = load_model()

def classify_text(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

def read_pdf(file_path):
    pdf_text = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            full_text += page.extract_text() + "\n"

    questions = re.split(r'\n\d+\.', full_text)
    questions = [question.strip() for question in questions if question.strip()]
    return questions

def append_to_csv(file_path, lines, topics):
    df_new = pd.DataFrame({'Topic': topics})
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        # Ensure both dataframes have the same number of rows
        assert len(df_existing) == len(df_new), "Both dataframes must have the same number of rows"
        # Concatenate the new topics to the existing dataframe
        df_combined = pd.concat([df_existing, df_new['Topic']], axis=1)
        return df_combined
    else:
        return df_new

def read_csv(file_path):
    return pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()

def generate_charts(df):
    # Ensure the chart folder exists
    if not os.path.exists(app.config["CHART_FOLDER"]):
        os.makedirs(app.config["CHART_FOLDER"])
    
    # Generate bar plot for marks distribution
    marks_chart_path = os.path.join(app.config["CHART_FOLDER"], 'marks_distribution.png')
    df.set_index('Question', inplace=True)
    df.iloc[:, :4].plot(kind='bar', figsize=(10, 6))
    plt.title('Marks Distribution for Each Question')
    plt.xlabel('Questions')
    plt.ylabel('Marks')
    plt.legend(title='Sections')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(marks_chart_path)
    plt.close()

    # Generate pie chart for topic distribution
    topic_chart_path = os.path.join(app.config["CHART_FOLDER"], 'topic_distribution.png')
    topic_counts = df['Topic'].value_counts()
    topic_counts.plot(kind='pie', figsize=(8, 8), autopct='%1.1f%%', startangle=140)
    plt.title('Topic Distribution')
    plt.ylabel('')
    plt.savefig(topic_chart_path)
    plt.close()

    # Generate bar plot for total marks by topic
    total_marks_by_topic_path = os.path.join(app.config["CHART_FOLDER"], 'total_marks_by_topic.png')
    total_marks_by_topic = df.groupby('Topic').sum().sum(axis=1)
    total_marks_by_topic.plot(kind='bar', figsize=(10, 6))
    plt.title('Total Marks by Topic')
    plt.xlabel('Topics')
    plt.ylabel('Total Marks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(total_marks_by_topic_path)
    plt.close()

    # Generate stacked bar chart for section performance by topic
    section_performance_by_topic_path = os.path.join(app.config["CHART_FOLDER"], 'section_performance_by_topic.png')
    section_performance = df.groupby('Topic').sum()
    section_performance.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Performance of Each Section for Each Topic')
    plt.xlabel('Topics')
    plt.ylabel('Total Marks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(section_performance_by_topic_path)
    plt.close()

    return marks_chart_path, topic_chart_path, total_marks_by_topic_path, section_performance_by_topic_path

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    login_form = LoginForm()
    signup_form = SignupForm()
    if login_form.validate_on_submit():
        email = login_form.email.data
        password = login_form.password.data
        user = users.get(email)
        if user and check_password_hash(user['password'], password):
            session['user'] = user
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    if signup_form.validate_on_submit():
        username = signup_form.username.data
        email = signup_form.email.data
        password = signup_form.password.data
        if email in users:
            flash('Email already exists', 'danger')
        else:
            users[email] = {'username': username, 'email': email, 'password': generate_password_hash(password)}
            session['user'] = users[email]
            flash('Account created successfully', 'success')
            return redirect(url_for('home'))
    return render_template('auth.html', login_form=login_form, signup_form=signup_form)

@app.route('/main', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect(url_for('auth'))
    form = UploadFileForm()
    filename = None
    pdf_content = []
    classified_topics = []
    marks_chart = None
    topic_chart = None
    total_marks_by_topic_chart = None
    section_performance_by_topic_chart = None
    if form.validate_on_submit():
        file = form.file.data
        if file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_content = read_pdf(file_path)
            pdf_content = [line for line in pdf_content if line.strip()]
            classified_topics = [classify_text(line) for line in pdf_content]
            csv_file_path = app.config["CSV_FILE_PATH"]
            combined_df = append_to_csv(csv_file_path, pdf_content, classified_topics)
            new_csv_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'marks_topic_2.csv')
            if os.path.exists(new_csv_filename):
                os.remove(new_csv_filename)
            combined_df.to_csv(new_csv_filename, index=False)

            # Generate charts
            marks_chart, topic_chart, total_marks_by_topic_chart, section_performance_by_topic_chart = generate_charts(combined_df)
            marks_chart = url_for('static', filename='charts/marks_distribution.png')
            topic_chart = url_for('static', filename='charts/topic_distribution.png')
            total_marks_by_topic_chart = url_for('static', filename='charts/total_marks_by_topic.png')
            section_performance_by_topic_chart = url_for('static', filename='charts/section_performance_by_topic.png')
    return render_template('index.html', form=form, filename=filename, pdf_content=pdf_content, classified_topics=classified_topics, marks_chart=marks_chart, topic_chart=topic_chart, total_marks_by_topic_chart=total_marks_by_topic_chart, section_performance_by_topic_chart=section_performance_by_topic_chart)

if __name__ == '__main__':
    app.run(debug=True)
