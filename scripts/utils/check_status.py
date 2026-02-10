#!/usr/bin/env python3
"""
ShikshaSetu System Status Checker
Complete health check for all system components
"""

import sys
import subprocess
import requests
from pathlib import Path

def check_service(name: str, check_func) -> bool:
    """Check a service and print status"""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name} - NOT RUNNING")
            return False
    except Exception as e:
        print(f"❌ {name} - ERROR: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("SHIKSHA SETU SYSTEM STATUS CHECK")
    print("="*60 + "\n")
    
    checks_passed = 0
    total_checks = 0
    
    # Backend Check
    total_checks += 1
    def check_backend():
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    
    if check_service("Backend API (Port 8000)", check_backend):
        checks_passed += 1
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            data = response.json()
            print(f"   Status: {data.get('status')}")
        except:
            pass
    
    # Frontend Check
    total_checks += 1
    def check_frontend():
        response = requests.get("http://localhost:5173", timeout=5)
        return response.status_code == 200
    
    if check_service("Frontend (Port 5173)", check_frontend):
        checks_passed += 1
    
    # Redis Check
    total_checks += 1
    def check_redis():
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=2)
            return r.ping()
        except:
            return False
    
    if check_service("Redis Server (Port 6379)", check_redis):
        checks_passed += 1
    
    # PostgreSQL Check
    total_checks += 1
    def check_postgres():
        try:
            from sqlalchemy import create_engine
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                return False
            
            engine = create_engine(db_url, pool_pre_ping=True)
            conn = engine.connect()
            conn.close()
            return True
        except:
            return False
    
    if check_service("PostgreSQL Database", check_postgres):
        checks_passed += 1
    
    # Celery Workers Check
    total_checks += 1
    def check_celery():
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        celery_processes = [line for line in result.stdout.split('\n') 
                           if 'celery' in line.lower() and 'worker' in line.lower() 
                           and 'grep' not in line]
        return len(celery_processes) > 0
    
    if check_service("Celery Workers", check_celery):
        checks_passed += 1
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        worker_count = len([l for l in result.stdout.split('\n') 
                           if 'celery' in l.lower() and 'worker' in l.lower() 
                           and 'grep' not in l])
        print(f"   Active workers: {worker_count}")
    
    # File Structure Check
    total_checks += 1
    def check_files():
        required = [
            Path(".env"),
            Path("backend/api/async_app.py"),
            Path("frontend/src/App.tsx"),
            Path("data/uploads"),
            Path("logs")
        ]
        return all(p.exists() for p in required)
    
    if check_service("File Structure", check_files):
        checks_passed += 1
    
    print("\n" + "="*60)
    score = (checks_passed / total_checks * 100) if total_checks > 0 else 0
    print(f"SYSTEM STATUS: {checks_passed}/{total_checks} ({score:.0f}%)")
    print("="*60 + "\n")
    
    if score == 100:
        print("🎉 PERFECT! All systems are operational!")
        print("✅ Ready to serve requests at:")
        print("   🌐 http://localhost:5173")
        print("   🔧 http://localhost:8000/docs\n")
        return 0
    elif score >= 80:
        print("⚠️  PARTIAL - Some services are down")
        print("   Check logs for details\n")
        return 1
    else:
        print("❌ CRITICAL - Multiple services are down")
        print("   Run: ./start_services.sh\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
