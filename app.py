import os
import logging
from flask import Flask, render_template, request, jsonify,redirect, url_for, flash, session, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager
import pandas as pd
from datetime import datetime
from ml_predictor import StockPredictor
from extensions import db
from models import User, Upload, Prediction, Company, InvestmentPrediction, CompanyComparison
from concurrent.futures import ThreadPoolExecutor
from flask_caching import Cache
from utils import convert_numpy_types, serialize_prediction_data, generate_prediction_pdf
import time
import json
from seed_companies import initialize_companies
from dotenv import load_dotenv
load_dotenv()
from apscheduler.schedulers.background import BackgroundScheduler
from seed_companies import update_company_data  # make sure your update function is imported
import atexit
'''
import cohere
COHERE_API_KEY = "3tutYe8F70AhegSDZVURXK1mKkiQQFAIROGh0JoZ"   # Replace with your real key
co = cohere.Client(COHERE_API_KEY)
'''
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
predictor = StockPredictor()

app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key_for_development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL environment variable is required for persistent storage")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Adjust settings for SQLite
if database_url.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {}
else:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
    }

db.init_app(app)
with app.app_context():
    try:
        db.create_all()
        logging.info("Database tables created successfully")
        initialize_companies()
        logging.info("Predefined companies initialized")
        scheduler = BackgroundScheduler()
        # Schedule to run daily at midnight
        scheduler.add_job(func=lambda: update_company_data(), trigger="cron", hour=0, minute=0)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())
        logging.info("APScheduler started for daily company updates")
        # ----------------------------------------------------------------
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
executor = ThreadPoolExecutor(max_workers=4)

def generate_mock_historical_data(current_price, days=30):
    """Generate mock historical data for a company"""
    import random
    from datetime import datetime, timedelta
    
    data = []
    price = current_price * 0.9  # Start from 90% of current price
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        
        # Random price movement (±5%)
        change_percent = random.uniform(-0.05, 0.05)
        price = price * (1 + change_percent)
        
        # Ensure price doesn't go too far from current price
        if price > current_price * 1.2:
            price = current_price * 1.2
        elif price < current_price * 0.8:
            price = current_price * 0.8
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Close': round(price, 2),
            'High': round(price * 1.02, 2),
            'Low': round(price * 0.98, 2),
            'Open': round(price * 0.99, 2),
            'Volume': random.randint(100000, 1000000)
        })
    
    return data

def run_model(processed_input):
    time.sleep(3)  
    return {"prediction": 123.45}

# Preprocess function
def preprocess(data):
    # Example preprocessing
    return data

# Async job
def background_prediction(job_id, processed_input):
    result = run_model(processed_input)
    cache.set(job_id, result, timeout=300)  # store result for 5 min
'''  
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please type something."})

    try:
        # Use Cohere Chat API
        response = co.chat(
            model="command-r-plus",  # latest Cohere chat model
            message=f"You are a helpful stock market assistant. User asked: {user_message}",
            temperature=0.7
        )

        bot_reply = response.text.strip()
        return jsonify({"response": bot_reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": f"Error: {str(e)}"})
'''
@app.route('/get_result', methods=['POST'])
def get_result():
    input_data = request.json
    cache_key = str(input_data)
    result = cache.get(cache_key)
    if result:
        return jsonify({"status": "completed", "result": result})
    return jsonify({"status": "pending"})

def format_inr_currency(amount):
    """Format amount as Indian Rupees with commas"""
    if amount is None or amount == '':
        return "₹0.00"
    try:
        return f"₹{float(amount):,.2f}"
    except (ValueError, TypeError):
        return "₹0.00"

app.jinja_env.filters['inr'] = format_inr_currency

def login_required(f):
    """Decorator to require login for certain routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
  

@app.route('/')
def index():
    """Home page - redirect to login"""
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            flash('Please enter both email and password.', 'danger')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            # Try to register new user if login fails
            if not user:
                try:
                    # Auto-register new user
                    username = email.split('@')[0]  # Use email prefix as username
                    # Make username unique if it already exists
                    counter = 1
                    original_username = username
                    while User.query.filter_by(username=username).first():
                        username = f"{original_username}{counter}"
                        counter += 1
                    
                    password_hash = generate_password_hash(password)
                    new_user = User(username=username, email=email, password_hash=password_hash)
                    db.session.add(new_user)
                    db.session.commit()
                    
                    session['user_id'] = new_user.id
                    session['username'] = new_user.username
                    flash('Welcome! Your account has been created successfully.', 'success')
                    return redirect(url_for('dashboard'))
                except Exception as e:
                    app.logger.error(f"Error creating user: {e}")
                    flash('Error creating account. Please try again.', 'danger')
            else:
                flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route("/")
def home():
    return render_template("dashboard.html")  # Your dashboard file

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing prediction history"""
    try:
        user_id = session['user_id']
        
        # Get recent predictions for this user
        recent_predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).limit(5).all()
        
        # Get recent investment predictions
        recent_investments = InvestmentPrediction.query.filter_by(user_id=user_id).order_by(InvestmentPrediction.created_at.desc()).limit(5).all()
        
        # Get recent company comparisons
        recent_comparisons = CompanyComparison.query.filter_by(user_id=user_id).order_by(CompanyComparison.created_at.desc()).limit(5).all()
        
        # Process prediction data for template
        processed_predictions = []
        for prediction in recent_predictions:
            prediction_info = {
                'id': prediction.id,
                'created_at': prediction.created_at,
                'predicted_price': prediction.predicted_price,
                'confidence_score': prediction.confidence_score,
                'model_type': prediction.model_type,
                'upload': prediction.upload,
                'latest_price': None,
                'predicted_change': None,
                'predicted_change_percent': None
            }
            
            # Parse prediction data safely
            if prediction.prediction_data:
                try:
                    import ast
                    details = ast.literal_eval(prediction.prediction_data)
                    prediction_info['latest_price'] = details.get('latest_price', prediction.predicted_price * 0.95)
                    prediction_info['predicted_change'] = details.get('predicted_change', prediction.predicted_price - prediction_info['latest_price'])
                    prediction_info['predicted_change_percent'] = details.get('predicted_change_percent', 0)
                except:
                    # Fallback values if parsing fails
                    prediction_info['latest_price'] = prediction.predicted_price * 0.95
                    prediction_info['predicted_change'] = prediction.predicted_price - prediction_info['latest_price']
                    prediction_info['predicted_change_percent'] = (prediction_info['predicted_change'] / prediction_info['latest_price'] * 100)
            else:
                # Fallback values if no prediction data
                prediction_info['latest_price'] = prediction.predicted_price * 0.95
                prediction_info['predicted_change'] = prediction.predicted_price - prediction_info['latest_price']
                prediction_info['predicted_change_percent'] = (prediction_info['predicted_change'] / prediction_info['latest_price'] * 100)
            
            processed_predictions.append(prediction_info)
        
        # Get upload history
        uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.created_at.desc()).limit(5).all()
        
        app.logger.info(f"Dashboard loaded for user {user_id} with {len(processed_predictions)} predictions")
        
        return render_template('dashboard.html', 
                             predictions=processed_predictions, 
                             uploads=uploads,
                             investment_predictions=recent_investments,
                             company_comparisons=recent_comparisons)
        
    except Exception as e:
        app.logger.error(f"Dashboard error for user {session.get('user_id')}: {e}")
        flash('Error loading dashboard. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handle CSV file upload and prediction"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected.', 'danger')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected.', 'danger')
                return redirect(request.url)
            
            if file and file.filename:
                # Accept any file extension but validate content
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                app.logger.info(f"Saving file to: {filepath}")
                file.save(filepath)
                
                # Verify file exists and is readable
                if not os.path.exists(filepath):
                    flash('Error saving file. Please try again.', 'danger')
                    return redirect(request.url)
                
                # Get file size for storage
                file_size = os.path.getsize(filepath)
                
                # Test CSV reading with flexible parsing
                try:
                    # Try different separators and encodings
                    df = None
                    for sep in [',', ';', '\t']:
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                df = pd.read_csv(filepath, sep=sep, encoding=encoding, nrows=5)
                                if len(df.columns) > 1:  # Valid CSV structure
                                    break
                            except:
                                continue
                        if df is not None and len(df.columns) > 1:
                            break
                    
                    if df is None or len(df.columns) <= 1:
                        flash('Unable to parse file. Please ensure it\'s a valid CSV format.', 'danger')
                        os.remove(filepath)
                        return redirect(request.url)
                    
                    # Re-read full file with correct parameters
                    df = pd.read_csv(filepath, sep=sep, encoding=encoding)
                    app.logger.info(f"CSV loaded successfully with {len(df)} rows and columns: {list(df.columns)}")
                    
                except Exception as csv_error:
                    flash(f'Invalid file format: {str(csv_error)}', 'danger')
                    app.logger.error(f"CSV reading error: {csv_error}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(request.url)
                
                # Record the upload
                upload = Upload()
                upload.user_id = session['user_id']
                upload.filename = filename
                upload.original_filename = file.filename
                upload.file_size = file_size
                db.session.add(upload)
                db.session.commit()
                app.logger.info(f"Upload recorded with ID: {upload.id}")
                
                # Process the file and make predictions
                app.logger.info("Starting prediction process...")
                predictor = StockPredictor()
                results = predictor.predict_from_csv(filepath)
                app.logger.info(f"Prediction results: {results.get('success', False)}")
                
                if results['success']:
                    # Convert and serialize prediction results properly
                    converted_results = convert_numpy_types(results)
                    
                    # Save prediction results
                    prediction = Prediction()
                    prediction.user_id = session['user_id']
                    prediction.upload_id = upload.id
                    prediction.predicted_price = converted_results['predicted_price']
                    prediction.confidence_score = converted_results['confidence_score']
                    prediction.model_type = converted_results['model_type']
                    prediction.prediction_data = serialize_prediction_data(converted_results['prediction_details'])
                    db.session.add(prediction)
                    db.session.commit()
                    app.logger.info(f"Prediction saved with ID: {prediction.id}")
                    
                    flash('File uploaded and prediction completed successfully!', 'success')
                    return render_template('prediction_results.html', results=converted_results, upload=upload)
                else:
                    flash(f'Prediction failed: {results.get("error", "Unknown error")}', 'danger')
                    return redirect(request.url)
                    
        except Exception as e:
            app.logger.error(f"Upload error: {e}")
            flash(f'Upload failed: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/investment_prediction', methods=['GET', 'POST'])
@login_required
def investment_prediction():
    """Handle investment profitability prediction"""
    companies = Company.query.all()
    if request.method == 'POST':
        try:
            company_id = request.form.get('company_id')
            investment_amount = float(request.form.get('investment_amount', 0))
            time_horizon = request.form.get('time_horizon', '1_month')
            
            if not company_id or investment_amount <= 0:
                flash('Please select a company and enter a valid investment amount.', 'danger')
                return redirect(request.url)
            
            company = Company.query.get(company_id)
            if not company:
                flash('Invalid company selection.', 'danger')
                return redirect(request.url)
            
            # Prepare company data for prediction
            company_data = {
                'name': company.name,
                'current_price': company.current_price,
                'market_cap': company.market_cap,
                'pe_ratio': company.pe_ratio,
                'sector': company.sector,
                'historical_data': company.historical_data
            }
            
            # Make prediction
            predictor = StockPredictor()
            results = predictor.predict_investment_profitability(
                company_data, investment_amount, time_horizon
            )
            
            if results['success']:
                # Convert and serialize prediction results properly
                converted_results = convert_numpy_types(results)
                
                # Save prediction results
                investment_pred = InvestmentPrediction()
                investment_pred.user_id = session['user_id']
                investment_pred.company_id = company.id
                investment_pred.investment_amount = converted_results['investment_amount']
                investment_pred.predicted_return = converted_results['predicted_return']
                investment_pred.predicted_profit = converted_results['predicted_profit']
                investment_pred.profit_percentage = converted_results['profit_percentage']
                investment_pred.is_profitable = converted_results['is_profitable']
                investment_pred.confidence_score = converted_results['confidence_score']
                investment_pred.time_horizon = converted_results['time_horizon']
                investment_pred.prediction_details = serialize_prediction_data(converted_results['prediction_details'])
                
                db.session.add(investment_pred)
                db.session.commit()
                
                flash('Investment prediction completed successfully!', 'success')
                return render_template('investment_prediction.html', 
                       companies=companies, 
                       results=converted_results, 
                       company=company,
                       investment_amount=investment_amount)

            else:
                flash(f'Prediction failed: {results.get("error", "Unknown error")}', 'danger')
                return redirect(request.url)
                
        except ValueError:
            flash('Please enter a valid investment amount.', 'danger')
            return redirect(request.url)
        except Exception as e:
            app.logger.error(f"Investment prediction error: {e}")
            flash(f'Prediction failed: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('investment_prediction.html', companies=companies)

@app.route('/company_comparison', methods=['GET', 'POST'])
@login_required
def company_comparison():
    """Handle company comparison feature"""
    companies = Company.query.all()
    
    if request.method == 'POST':
        try:
            company1_id = request.form.get('company1_id')
            company2_id = request.form.get('company2_id')
            time_horizon = request.form.get('time_horizon', '1_month')
            
            if not company1_id or not company2_id:
                flash('Please select both companies for comparison.', 'danger')
                return redirect(request.url)
            
            if company1_id == company2_id:
                flash('Please select two different companies for comparison.', 'danger')
                return redirect(request.url)
            
            company1 = Company.query.get(company1_id)
            company2 = Company.query.get(company2_id)
            
            if not company1 or not company2:
                flash('Invalid company selection.', 'danger')
                return redirect(request.url)
            
            # Prepare company data for comparison
            company1_data = {
                'name': company1.name,
                'current_price': company1.current_price,
                'market_cap': company1.market_cap,
                'pe_ratio': company1.pe_ratio,
                'sector': company1.sector,
                'historical_data': company1.historical_data
            }
            
            company2_data = {
                'name': company2.name,
                'current_price': company2.current_price,
                'market_cap': company2.market_cap,
                'pe_ratio': company2.pe_ratio,
                'sector': company2.sector,
                'historical_data': company2.historical_data
            }
            
            # Make comparison
            predictor = StockPredictor()
            results = predictor.compare_companies(company1_data, company2_data, time_horizon)
            
            if results['success']:
                # Convert and serialize comparison results properly
                converted_results = convert_numpy_types(results)
                
                # Save comparison results
                comparison = CompanyComparison()
                comparison.user_id = session['user_id']
                comparison.company1_id = company1.id
                comparison.company2_id = company2.id
                comparison.company1_predicted_return = converted_results['company1_predicted_return']
                comparison.company2_predicted_return = converted_results['company2_predicted_return']
                comparison.recommended_company_id = company1.id if converted_results['recommended_company'] == 1 else company2.id
                comparison.confidence_score = converted_results['confidence_score']
                comparison.time_horizon = converted_results['time_horizon']
                comparison.comparison_details = serialize_prediction_data(converted_results['comparison_details'])
                
                db.session.add(comparison)
                db.session.commit()
                
                flash('Company comparison completed successfully!', 'success')
                return render_template('company_comparison.html',
                                     companies=companies,
                                     results=converted_results,
                                     company1=company1,
                                     company2=company2)
            else:
                flash(f'Comparison failed: {results.get("error", "Unknown error")}', 'danger')
                return redirect(request.url)
                
        except Exception as e:
            app.logger.error(f"Company comparison error: {e}")
            flash(f'Comparison failed: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('company_comparison.html', companies=companies)

# PDF Download Routes
@app.route('/download_pdf/<prediction_type>/<int:record_id>')
@login_required
def download_prediction_pdf(prediction_type, record_id):
    """Generate and download PDF for prediction results"""
    try:
        user_id = session['user_id']
        
        if prediction_type == 'stock_prediction':
            prediction = Prediction.query.filter_by(id=record_id, user_id=user_id).first()
            if not prediction:
                flash('Prediction not found.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Prepare data for PDF
            pdf_data = {
                'predicted_price': prediction.predicted_price,
                'confidence_score': prediction.confidence_score,
                'model_type': prediction.model_type,
                'prediction_details': json.loads(prediction.prediction_data) if prediction.prediction_data else {}
            }
            filename = f'stock_prediction_{record_id}.pdf'
            
        elif prediction_type == 'investment_prediction':
            prediction = InvestmentPrediction.query.filter_by(id=record_id, user_id=user_id).first()
            if not prediction:
                flash('Investment prediction not found.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Prepare data for PDF
            pdf_data = {
                'investment_amount': prediction.investment_amount,
                'predicted_return': prediction.predicted_return,
                'predicted_profit': prediction.predicted_profit,
                'profit_percentage': prediction.profit_percentage,
                'is_profitable': prediction.is_profitable,
                'confidence_score': prediction.confidence_score,
                'time_horizon': prediction.time_horizon,
                'prediction_details': json.loads(prediction.prediction_details) if prediction.prediction_details else {}
            }
            filename = f'investment_prediction_{record_id}.pdf'
            
        elif prediction_type == 'company_comparison':
            comparison = CompanyComparison.query.filter_by(id=record_id, user_id=user_id).first()
            if not comparison:
                flash('Company comparison not found.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Prepare data for PDF
            pdf_data = {
                'company1_predicted_return': comparison.company1_predicted_return,
                'company2_predicted_return': comparison.company2_predicted_return,
                'recommended_company': 1 if comparison.recommended_company_id == comparison.company1_id else 2,
                'confidence_score': comparison.confidence_score,
                'time_horizon': comparison.time_horizon,
                'comparison_details': json.loads(comparison.comparison_details) if comparison.comparison_details else {}
            }
            filename = f'company_comparison_{record_id}.pdf'
        else:
            flash('Invalid prediction type.', 'danger')
            return redirect(url_for('dashboard'))
        
        # Generate PDF
        pdf_buffer = generate_prediction_pdf(pdf_data, prediction_type)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        flash('Error generating PDF. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

# Quick PDF download routes for results pages
@app.route('/download_current_prediction_pdf', methods=['POST'])
@login_required
def download_current_prediction_pdf():
    """Download PDF for current prediction results"""
    try:
        prediction_data = request.json
        prediction_type = prediction_data.get('type', 'stock_prediction')
        
        if not prediction_data:
            return jsonify({'error': 'No prediction data provided'}), 400
        
        # Generate PDF
        pdf_buffer = generate_prediction_pdf(prediction_data, prediction_type)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'{prediction_type}_results.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        return jsonify({'error': 'Error generating PDF'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
