from flask import Flask, render_template, request,redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
       # Here, you would handle saving data to the database.
        # For now, we'll skip straight to redirecting.
        return redirect(url_for('index'));
    return render_template('Login.html')
   

if __name__ == '__main__':
    app.run(debug=True)
