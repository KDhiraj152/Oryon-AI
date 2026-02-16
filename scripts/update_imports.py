#!/usr/bin/env python3
"""Update imports after folder reorganization."""
import glob
import os
import re

# Import path mappings (old -> new)
MAPPINGS = {
    # Cache
    'backend.cache': 'backend.infra.cache',
    # Hardware  
    'backend.hardware': 'backend.infra.hardware',
    # Telemetry
    'backend.telemetry': 'backend.infra.telemetry',
    # Execution -> runtime
    'backend.execution': 'backend.infra.runtime',
    # Orchestration -> runtime
    'backend.orchestration': 'backend.infra.runtime',
    # Scalability health -> infra
    'backend.scalability.health': 'backend.infra.health',
    'backend.scalability': 'backend.infra',
    # Services - inference -> ml
    'backend.services.inference': 'backend.ml.inference',
    # Services - pipeline -> ml
    'backend.services.pipeline': 'backend.ml.pipeline',
    # Services - tts -> ml/speech
    'backend.services.tts': 'backend.ml.speech.tts',
    # Services - speech_generator/processor -> ml/speech
    'backend.services.speech_generator': 'backend.ml.speech.speech_generator',
    'backend.services.speech_processor': 'backend.ml.speech.speech_processor',
    # Services - translate -> ml
    'backend.services.translate': 'backend.ml.translate',
    # Services - ocr -> ml
    'backend.services.ocr': 'backend.ml.ocr.ocr',
    # Services - ai_core -> services/chat
    'backend.services.ai_core': 'backend.services.chat',
    # Services - rag -> services/chat
    'backend.services.rag': 'backend.services.chat.rag',
    # Services - simplifier -> services/content
    'backend.services.simplifier': 'backend.services.content.simplifier',
    # Services - content_validation -> services/content
    'backend.services.content_validation': 'backend.services.content.content_validation',
    # Services - cultural_context -> services/content
    'backend.services.cultural_context': 'backend.services.content.cultural_context',
    # Services - grade_adaptation -> services/content
    'backend.services.grade_adaptation': 'backend.services.content.grade_adaptation',
    # Services - safety_pipeline -> services/content
    'backend.services.safety_pipeline': 'backend.services.content.safety_pipeline',
    # Services - validate -> services/content
    'backend.services.validate': 'backend.services.content.validate',
    # Services - user_profile -> services/users
    'backend.services.user_profile': 'backend.services.users.user_profile',
    # Services - review_queue -> services/users
    'backend.services.review_queue': 'backend.services.users.review_queue',
    # Services - evaluation -> ml
    'backend.services.evaluation': 'backend.ml.evaluation',
    # Middleware -> api/middleware
    'backend.middleware': 'backend.api.middleware',
    # Database
    'backend.database': 'backend.db.database',
}

def update_imports(content):
    """Update imports in a file content."""
    # Sort by length (longest first) to avoid partial replacements
    sorted_mappings = sorted(MAPPINGS.items(), key=lambda x: -len(x[0]))
    
    for old, new in sorted_mappings:
        # Match 'from X import' or 'import X'
        pattern = rf'\b{re.escape(old)}\b'
        content = re.sub(pattern, new, content)
    
    return content

def main():
    os.chdir('/Users/kdhiraj/Downloads/Oryon-setu')
    files = glob.glob('backend/**/*.py', recursive=True) + glob.glob('tests/**/*.py', recursive=True)
    files = [f for f in files if '__pycache__' not in f]
    
    changed = 0
    for filepath in files:
        with open(filepath) as f:
            content = f.read()
        
        new_content = update_imports(content)
        
        if content != new_content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f'Updated: {filepath}')
            changed += 1
    
    print(f'\nTotal files updated: {changed}')

if __name__ == '__main__':
    main()
