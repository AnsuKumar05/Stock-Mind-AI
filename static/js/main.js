/**
 * AI Stock Market Analyzer - Main JavaScript
 * Handles UI interactions, form validation, and dynamic content
 */

// Global variables
let currentUser = null;
let notifications = [];

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize Bootstrap components
    initializeBootstrap();
    
    // Setup form validations
    setupFormValidations();
    
    // Setup file upload handlers
    setupFileUploadHandlers();
    
    // Setup number formatting
    setupNumberFormatting();
    
    // Setup tooltips and animations
    setupUIEnhancements();
    
    // Setup chart initialization
    initializeCharts();
    
    // Setup notification system
    initializeNotifications();
    
    console.log('AI Stock Analyzer initialized successfully');
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrap() {
    // Initialize all tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize all popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    

/**
 * Setup form validations
 */
function setupFormValidations() {
    // Bootstrap form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Custom investment amount validation
    const investmentAmountInputs = document.querySelectorAll('input[name="investment_amount"]');
    investmentAmountInputs.forEach(function(input) {
        input.addEventListener('input', function() {
            validateInvestmentAmount(this);
        });
    });
    
    // Company selection validation
    const companySelects = document.querySelectorAll('select[name^="company"]');
    companySelects.forEach(function(select) {
        select.addEventListener('change', function() {
            validateCompanySelection();
        });
    });
}

/**
 * Validate investment amount
 */
function validateInvestmentAmount(input) {
    // Remove commas before parsing
    const value = parseFloat(input.value.replace(/,/g, ''));
    const min = 500;
    const max = 1000000000;

    let isValid = true;
    let message = '';

    if (isNaN(value) || value < min) {
        isValid = false;
        message = `Minimum investment amount is ₹${min.toLocaleString('en-IN')}`;
    } else if (value > max) {
        isValid = false;
        message = `Maximum investment amount is ₹${max.toLocaleString('en-IN')}`;
    }

    // Update validation state
    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }

    // Update feedback message
    let feedback = input.parentNode.querySelector('.invalid-feedback');
    if (!feedback) {
        feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        input.parentNode.appendChild(feedback);
    }
    feedback.textContent = message;

    return isValid;
}

/**
 * Validate company selection for comparison
 */
function validateCompanySelection() {
    const company1Select = document.getElementById('company1_id');
    const company2Select = document.getElementById('company2_id');
    
    if (!company1Select || !company2Select) return;
    
    const company1Value = company1Select.value;
    const company2Value = company2Select.value;
    
    // Check for same company selection
    if (company1Value && company2Value && company1Value === company2Value) {
        company2Select.classList.add('is-invalid');
        
        let feedback = company2Select.parentNode.querySelector('.invalid-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            company2Select.parentNode.appendChild(feedback);
        }
        feedback.textContent = 'Please select a different company for comparison';
        
        return false;
    } else {
        company2Select.classList.remove('is-invalid');
        return true;
    }
}

/**
 * Setup file upload handlers
 */
function setupFileUploadHandlers() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(function(input) {
        input.addEventListener('change', function(e) {
            handleFileSelection(e.target);
        });
        
        // Add drag and drop functionality
        const formGroup = input.closest('.mb-3, .mb-4');
        if (formGroup) {
            setupDragAndDrop(formGroup, input);
        }
    });
}

/**
 * Handle file selection
 */
function handleFileSelection(input) {
    const file = input.files[0];
    if (!file) return;
    
    // Validate file type
    const allowedTypes = ['.csv', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showNotification('Please select a CSV or TXT file', 'error');
        input.value = '';
        return;
    }
    
    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File size exceeds 50MB limit', 'error');
        input.value = '';
        return;
    }
    
    // Show file info
    updateFileInfo(input, file);
    
    // Enable form submission
    const form = input.closest('form');
    if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = false;
        }
    }
}

/**
 * Update file information display
 */
function updateFileInfo(input, file) {
    let fileInfo = input.parentNode.querySelector('.file-info');
    
    if (!fileInfo) {
        fileInfo = document.createElement('div');
        fileInfo.className = 'file-info mt-2 p-2 bg-light rounded';
        input.parentNode.appendChild(fileInfo);
    }
    
    const fileSize = (file.size / (1024 * 1024)).toFixed(2);
    fileInfo.innerHTML = `
        <small class="text-muted">
            <i class="fas fa-file-csv me-1"></i>
            <strong>${file.name}</strong> (${fileSize} MB)
            <span class="text-success ms-2">
                <i class="fas fa-check-circle me-1"></i>Ready to upload
            </span>
        </small>
    `;
}

/**
 * Setup drag and drop functionality
 */
function setupDragAndDrop(container, input) {
    container.classList.add('drag-drop-area');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        container.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, unhighlight, false);
    });
    
    container.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            input.files = files;
            handleFileSelection(input);
        }
    }, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        container.classList.add('drag-over');
    }
    
    function unhighlight(e) {
        container.classList.remove('drag-over');
    }
}

/**
 * Setup number formatting
 */
function validateInvestmentAmount(input) {
    // Remove commas before parsing
    const value = parseFloat(input.value.replace(/,/g, ''));
    const min = 500;
    const max = 1000000000;

    let isValid = true;
    let message = '';

    if (isNaN(value) || value < min) {
        isValid = false;
        message = `Minimum investment amount is ₹${min.toLocaleString('en-IN')}`;
    } else if (value > max) {
        isValid = false;
        message = `Maximum investment amount is ₹${max.toLocaleString('en-IN')}`;
    }

    // Update validation state
    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }

    // Update feedback message
    let feedback = input.parentNode.querySelector('.invalid-feedback');
    if (!feedback) {
        feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        input.parentNode.appendChild(feedback);
    }
    feedback.textContent = message;

    return isValid;
}
/**
 * Format currency input
 */
function formatCurrencyInput(input) {
    const value = parseFloat(input.value.replace(/[^\d.-]/g, ''));
    if (!isNaN(value)) {
        input.value = value.toLocaleString('en-IN');
    }
}

/**
 * Unformat currency input for editing
 */
function unformatCurrencyInput(input) {
    const value = input.value.replace(/[^\d.-]/g, '');
    input.value = value;
}

/**
 * Format display numbers
 */
function formatDisplayNumbers() {
    const numberElements = document.querySelectorAll('[data-format="number"]');
    numberElements.forEach(function(element) {
        const value = parseFloat(element.textContent);
        if (!isNaN(value)) {
            element.textContent = value.toLocaleString('en-IN');
        }
    });
    
    const currencyElements = document.querySelectorAll('[data-format="currency"]');
    currencyElements.forEach(function(element) {
        const value = parseFloat(element.textContent.replace(/[^\d.-]/g, ''));
        if (!isNaN(value)) {
            element.textContent = '₹' + value.toLocaleString('en-IN', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
    });
}

/**
 * Setup UI enhancements
 */
function setupUIEnhancements() {
    // Add loading states to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                addLoadingState(submitBtn);
            }
        });
    });
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(function(card, index) {
        card.style.animationDelay = (index * 0.1) + 's';
        card.classList.add('fade-in');
    });
    
    // Setup smooth scrolling
    const scrollLinks = document.querySelectorAll('a[href^="#"]');
    scrollLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Add loading state to button
 */
function addLoadingState(button) {
    const originalText = button.innerHTML;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        Processing...
    `;
    button.disabled = true;
    
    // Store original text for potential restoration
    button.dataset.originalText = originalText;
}

/**
 * Remove loading state from button
 */
function removeLoadingState(button) {
    if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        button.disabled = false;
    }
}

/**
 * Initialize charts
 */
function initializeCharts() {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded');
        return;
    }
    
    // Default chart configuration
    Chart.defaults.color = '#e9ecef';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.125)';
    Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.05)';
    
    // Initialize prediction charts
    initializePredictionCharts();
    
    // Initialize comparison charts
    initializeComparisonCharts();
}

/**
 * Initialize prediction charts
 */
function initializePredictionCharts() {
    const chartContainer = document.getElementById('predictionChart');
    if (!chartContainer) return;
    
    const ctx = chartContainer.getContext('2d');
    
    // Sample data - in real implementation, this would come from the server
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
            datasets: [{
                label: 'Predicted Price',
                data: [100, 102, 98, 105, 107, 103, 110],
                borderColor: 'rgb(13, 110, 253)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

/**
 * Initialize comparison charts
 */
function initializeComparisonCharts() {
    const chartContainer = document.getElementById('comparisonChart');
    if (!chartContainer) return;
    
    const ctx = chartContainer.getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Return Potential', 'Risk Level', 'Market Cap', 'P/E Ratio', 'Volatility', 'Confidence'],
            datasets: [{
                label: 'Company A',
                data: [8, 6, 9, 7, 5, 8],
                borderColor: 'rgb(13, 110, 253)',
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
            }, {
                label: 'Company B',
                data: [6, 8, 7, 8, 6, 7],
                borderColor: 'rgb(25, 135, 84)',
                backgroundColor: 'rgba(25, 135, 84, 0.2)',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10
                }
            }
        }
    });
}

/**
 * Initialize notification system
 */
function initializeNotifications() {
    // Create notification container if it doesn't exist
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    const id = 'notification-' + Date.now();
    const typeClass = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'warning': 'bg-warning',
        'info': 'bg-info'
    }[type] || 'bg-info';
    
    const notification = document.createElement('div');
    notification.id = id;
    notification.className = `toast align-items-center text-white ${typeClass} border-0`;
    notification.setAttribute('role', 'alert');
    notification.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${getIconForType(type)} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    container.appendChild(notification);
    
    const toast = new bootstrap.Toast(notification, {
        delay: duration
    });
    toast.show();
    
    // Remove from DOM after hiding
    notification.addEventListener('hidden.bs.toast', function() {
        notification.remove();
    });
    
    return id;
}

/**
 * Get icon for notification type
 */
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Utility functions
 */

// Format number as Indian currency
function formatINR(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR'
    }).format(amount);
}

// Format percentage
function formatPercentage(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for performance
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Animate numbers (counting effect)
function animateNumber(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard!', 'success');
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showNotification('Copied to clipboard!', 'success');
            return true;
        } catch (err) {
            showNotification('Failed to copy to clipboard', 'error');
            return false;
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

// Export functions for global access
window.AIStockAnalyzer = {
    showNotification,
    formatINR,
    formatPercentage,
    copyToClipboard,
    animateNumber,
    validateInvestmentAmount,
    validateCompanySelection
};

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();

    // Attach the predict button handler AFTER the form exists
    const form = document.getElementById("investmentForm");
    if (form) {
        form.addEventListener("submit", onPredictClick);
        console.log(" Predict form listener attached");
    } else {
        console.error("InvestmentForm not found");
    }
});

console.log('AI Stock Analyzer JavaScript loaded successfully');

