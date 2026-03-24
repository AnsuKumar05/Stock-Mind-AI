from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from datetime import datetime, timedelta

def init_scheduler(app):
    """
    Initialize background scheduler. Passes app instance.
    """
    scheduler = BackgroundScheduler()
    
    # Add app reference context
    def run_hourly():
        from routes.data_fetcher import fetch_and_update_hourly
        with app.app_context():
            fetch_and_update_hourly(app)

    def run_cleanup():
        from routes.models import DailyStockData
        from routes.extensions import db
        with app.app_context():
            app.logger.info("Running 4-hour retention cleanup job.")
            cutoff = datetime.utcnow() - timedelta(hours=4)
            DailyStockData.query.filter(DailyStockData.timestamp < cutoff).delete()
            db.session.commit()

    scheduler.add_job(func=run_hourly, trigger="interval", hours=1, id="hourly_fetch_job", replace_existing=True, next_run_time=datetime.utcnow(), misfire_grace_time=3600)
    scheduler.add_job(func=run_cleanup, trigger="interval", hours=4, id="retention_cleanup", replace_existing=True, misfire_grace_time=3600)

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
