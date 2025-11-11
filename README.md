
## Overview

This is a Flask-based AI-powered stock market analysis web application that allows users to upload CSV files containing historical stock data and receive machine learning-based price predictions. The system provides comprehensive investment analysis, company comparisons, and profitability predictions through multiple ML models including Linear Regression, Ridge, Lasso, and Random Forest algorithms.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 with Flask for server-side rendering
- **UI Framework**: Bootstrap 5 with dark theme implementation
- **Styling**: Custom CSS with CSS variables and gradient backgrounds
- **Icons**: Font Awesome 6.4.0 for consistent iconography
- **JavaScript**: Vanilla JavaScript for form validation, file uploads, and UI interactions
- **Responsive Design**: Mobile-first approach using Bootstrap's grid system

### Backend Architecture
- **Web Framework**: Flask with ProxyFix middleware for production deployment
- **Database ORM**: SQLAlchemy with declarative base pattern
- **Authentication**: Session-based authentication with Werkzeug password hashing
- **File Processing**: Secure file upload handling with 50MB size limit
- **Concurrent Processing**: ThreadPoolExecutor for handling multiple ML predictions
- **Caching**: Flask-Caching for performance optimization
- **Performance Optimization**: Numba JIT compilation for fast technical indicator calculations

### Machine Learning Pipeline
- **ML Framework**: scikit-learn with multiple regression models
- **Model Selection**: Automatic cross-validation to choose best performing model
- **Feature Engineering**: Technical indicators including SMA, RSI, MACD calculations
- **Data Processing**: Pandas for CSV parsing with flexible column mapping
- **Preprocessing**: RobustScaler for outlier-resistant normalization
- **Prediction Types**: Single stock prediction, investment profitability analysis, and company comparison

### Database Architecture
- **Primary Database**: SQLAlchemy-compatible (SQLite for development, PostgreSQL for production)
- **Connection Pooling**: Configurable pool settings for production databases
- **Models**: User management, file uploads tracking, prediction results storage, company data, and comparison results
- **Data Integrity**: Foreign key relationships with cascade delete operations
- **Indexing**: Strategic indexing on frequently queried columns for performance

## External Dependencies

### Core Dependencies
- **Flask**: Web framework with SQLAlchemy ORM integration
- **scikit-learn**: Machine learning algorithms and preprocessing tools
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing foundation
- **numba**: JIT compilation for performance-critical calculations

### Production Dependencies
- **waitress**: Production WSGI server for deployment
- **python-dotenv**: Environment variable management
- **Flask-Caching**: Application-level caching system
- **Werkzeug**: Security utilities and file handling

### Frontend Dependencies
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements
- **Custom CSS**: Application-specific styling with dark theme

### Optional Dependencies
- **pdfkit**: PDF generation capabilities for reports
- **Flask-Login**: Enhanced authentication features (referenced but not fully implemented)

### Environment Variables
- **DATABASE_URL**: Database connection string (required)
- **SESSION_SECRET**: Flask session encryption key (fallback provided)

### Database Support
- **Development**: SQLite with simplified configuration
- **Production**: PostgreSQL with connection pooling and performance optimizations
- **Connection Management**: Automatic pool recycling and health checks for production environments