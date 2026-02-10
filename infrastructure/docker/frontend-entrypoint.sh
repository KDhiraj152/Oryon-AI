#!/bin/sh
# Frontend environment configuration injection

set -e

# Replace environment variables in runtime config if it exists
if [ -f /usr/share/nginx/html/config.js ]; then
    echo "Injecting environment variables into frontend config..."
    
    # Replace API_URL if set
    if [ -n "$API_URL" ]; then
        sed -i "s|API_URL:.*|API_URL: '$API_URL',|g" /usr/share/nginx/html/config.js
    fi
    
    # Replace other environment variables as needed
    if [ -n "$SENTRY_DSN" ]; then
        sed -i "s|SENTRY_DSN:.*|SENTRY_DSN: '$SENTRY_DSN',|g" /usr/share/nginx/html/config.js
    fi
    
    echo "Environment variables injected successfully"
fi

exit 0
