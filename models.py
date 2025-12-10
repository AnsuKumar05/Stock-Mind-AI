from extensions import db
from datetime import datetime
import json

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    uploads = db.relationship('Upload', backref='user', lazy=True, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    investment_predictions = db.relationship('InvestmentPrediction', backref='user', lazy=True, cascade='all, delete-orphan')
    company_comparisons = db.relationship('CompanyComparison', backref='user', lazy=True, cascade='all, delete-orphan')

class Upload(db.Model):
    """Model to track uploaded CSV files"""
    __tablename__ = 'uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=True)  # Support larger files up to 50MB
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='upload', lazy=True, cascade='all, delete-orphan')

class Prediction(db.Model):
    """Model to store prediction results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False, index=True)
    predicted_price = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float)
    model_type = db.Column(db.String(50), default='Enhanced Linear Regression')
    prediction_data = db.Column(db.Text)  # JSON string of detailed prediction data
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class Company(db.Model):
    """Model to store predefined companies for investment prediction"""
    __tablename__ = 'companies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    symbol = db.Column(db.String(10), nullable=False, unique=True)
    sector = db.Column(db.String(50), nullable=False)
    current_price = db.Column(db.Float, nullable=False)
    market_cap = db.Column(db.BigInteger, nullable=True)
    pe_ratio = db.Column(db.Float, nullable=True)
    historical_data = db.Column(db.Text)  # JSON string of historical price data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    investment_predictions = db.relationship('InvestmentPrediction', backref='company', lazy=True, cascade='all, delete-orphan')
    comparison_company1 = db.relationship('CompanyComparison', foreign_keys='CompanyComparison.company1_id', backref='company1', lazy=True, cascade='all, delete-orphan')
    comparison_company2 = db.relationship('CompanyComparison', foreign_keys='CompanyComparison.company2_id', backref='company2', lazy=True, cascade='all, delete-orphan')

class InvestmentPrediction(db.Model):
    """Model to store investment profitability predictions"""
    __tablename__ = 'investment_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False, index=True)
    investment_amount = db.Column(db.Float, nullable=False)
    predicted_return = db.Column(db.Float, nullable=False)
    predicted_profit = db.Column(db.Float, nullable=False)
    profit_percentage = db.Column(db.Float, nullable=False)
    is_profitable = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    time_horizon = db.Column(db.String(20), default='1_month')  # 1_week, 1_month, 1_year
    prediction_details = db.Column(db.Text)  # JSON string of detailed prediction data
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class CompanyComparison(db.Model):
    """Model to store company comparison results"""
    __tablename__ = 'company_comparisons'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    company1_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False, index=True)
    company2_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False, index=True)
    company1_predicted_return = db.Column(db.Float, nullable=False)
    company2_predicted_return = db.Column(db.Float, nullable=False)
    recommended_company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    time_horizon = db.Column(db.String(20), default='1_month')
    comparison_details = db.Column(db.Text)  # JSON string of detailed comparison data
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class ChatSession(db.Model):
    """Model to group chatbot conversations per user"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(255), default="New Chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')


class ChatMessage(db.Model):
    """Model to store chatbot messages"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False, index=True)
    sender = db.Column(db.String(20), nullable=False)  # 'user' or 'bot'
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
