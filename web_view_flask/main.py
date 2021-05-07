from flask import Blueprint, render_template, redirect, url_for,request
from flask_login import login_required, current_user
from collect_data import create_dataset
from collect_data import mysql_connector

main = Blueprint('main', __name__)
ticker='AAPL'
@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@main.route('/ticker')
@login_required
def ticker():
    return render_template('ticker.html', name='Choose a Stock you want to analyse!')

@main.route('/dashboard', methods=['POST', 'GET'])
@login_required
def dashboard():
    ticker = request.form.get('ticker')
    ticker = str(ticker)
    print(ticker)
    mysql_connector.update_mysql(ticker)
    return render_template('dashboard.html', name='Submitted!')