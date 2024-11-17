from flask import Flask, render_template

# Create the Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for the about me page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the Vehile Detection page
@app.route('/vedec')
def vedec():
    return render_template('vedec.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)