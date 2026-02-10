#!/bin/bash
# Setup HuggingFace Authentication for Gated Models
# This script helps configure HuggingFace authentication to access gated models

echo "=========================================="
echo "HuggingFace Authentication Setup"
echo "=========================================="
echo ""

echo "This script will help you authenticate with HuggingFace to access gated models."
echo "Gated models requiring authentication:"
echo "  • ai4bharat/indic-bert (Indian language BERT)"
echo "  • Other restricted models"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

echo "Choose authentication method:"
echo "  1. Login via CLI (recommended)"
echo "  2. Set environment variable"
echo "  3. Skip (use ungated alternatives)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📝 You'll need a HuggingFace token:"
        echo "   1. Go to: https://huggingface.co/settings/tokens"
        echo "   2. Create a new token (read access)"
        echo "   3. Paste it when prompted below"
        echo ""
        huggingface-cli login

        if [ $? -eq 0 ]; then
            echo "✅ Successfully authenticated with HuggingFace!"
            echo ""
            echo "🔓 Requesting access to gated models:"
            echo "   Visit: https://huggingface.co/ai4bharat/indic-bert"
            echo "   Click 'Request Access' and wait for approval"
        else
            echo "❌ Authentication failed"
            exit 1
        fi
        ;;
    2)
        echo ""
        read -p "Enter your HuggingFace token: " token
        if [ -z "$token" ]; then
            echo "❌ No token provided"
            exit 1
        fi

        # Add to .env file
        ENV_FILE=".env"
        if [ -f "$ENV_FILE" ]; then
            # Check if HUGGINGFACE_API_KEY already exists
            if grep -q "HUGGINGFACE_API_KEY" "$ENV_FILE"; then
                # Update existing
                sed -i.bak "s/HUGGINGFACE_API_KEY=.*/HUGGINGFACE_API_KEY=$token/" "$ENV_FILE"
                echo "✅ Updated HUGGINGFACE_API_KEY in $ENV_FILE"
            else
                # Add new
                echo "" >> "$ENV_FILE"
                echo "# HuggingFace Authentication" >> "$ENV_FILE"
                echo "HUGGINGFACE_API_KEY=$token" >> "$ENV_FILE"
                echo "✅ Added HUGGINGFACE_API_KEY to $ENV_FILE"
            fi
        else
            # Create new .env
            echo "# HuggingFace Authentication" > "$ENV_FILE"
            echo "HUGGINGFACE_API_KEY=$token" >> "$ENV_FILE"
            echo "✅ Created $ENV_FILE with HUGGINGFACE_API_KEY"
        fi

        echo ""
        echo "🔓 Don't forget to request access to gated models:"
        echo "   Visit: https://huggingface.co/ai4bharat/indic-bert"
        echo "   Click 'Request Access'"
        ;;
    3)
        echo ""
        echo "⏭️  Skipping authentication"
        echo "📝 Note: The system will use ungated alternatives:"
        echo "   • google/gemma-2-2b-it (primary validator)"
        echo "   • Full accuracy for Indian language content"
        echo ""
        echo "To enable gated models later, run this script again or:"
        echo "   huggingface-cli login"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test your setup:"
echo "     python scripts/testing/test_all_features.py"
echo ""
echo "  2. If using gated models, wait for access approval from:"
echo "     https://huggingface.co/ai4bharat/indic-bert"
echo ""
echo "  3. For Qwen local inference (optional):"
echo "     • Requires ~14GB disk space (FP16 model)"
echo "     • Works without quantization on macOS"
echo "     • Or use API mode with HUGGINGFACE_API_KEY"
echo ""
