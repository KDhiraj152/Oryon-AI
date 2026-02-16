#!/bin/bash
# Integration test script for frontend-backend communication

echo "🔍 Oryon AI Integration Test"
echo "================================"
echo ""

# Check if backend is running
echo "1. Checking backend health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ Backend is running (V2 API)"
else
    echo "   ❌ Backend is NOT running"
    echo "   Run: cd /Users/kdhiraj_152/Downloads/oryon && ./start.sh"
    exit 1
fi

# Check if frontend is running
echo ""
echo "2. Checking frontend dev server..."
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "   ✅ Frontend is running"
else
    echo "   ❌ Frontend is NOT running"
    echo "   Run: cd /Users/kdhiraj_152/Downloads/oryon/frontend && npm run dev"
    exit 1
fi

# Test API endpoints
echo ""
echo "3. Testing API endpoints..."

# Health check
echo "   - Health: "
HEALTH=$(curl -s http://localhost:8000/health)
echo "     $HEALTH"

# Guest chat endpoint (no auth required)
echo "   - Guest Chat (no auth): "
CHAT=$(curl -s -w "%{http_code}" -o /dev/null -X POST http://localhost:8000/api/v2/chat/guest \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "language": "en", "complexity_level": 5}')
if [ "$CHAT" = "200" ]; then
    echo "     ✅ Returns 200 (working)"
else
    echo "     Status: $CHAT"
fi

# Test proxy
echo ""
echo "4. Testing Vite proxy..."
PROXY=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:5173/health)
if [ "$PROXY" = "200" ]; then
    echo "   ✅ Proxy working - /health routes to backend"
else
    echo "   ❌ Proxy not working (status: $PROXY)"
fi

echo ""
echo "================================"
echo "Integration check complete!"
echo ""
echo "To fully test:"
echo "1. Open http://localhost:5173 in browser"
echo "2. Click 'Get Started' or 'Sign In'"
echo "3. Register a new account or use demo login"
echo "4. Try sending a message in chat"
echo ""
