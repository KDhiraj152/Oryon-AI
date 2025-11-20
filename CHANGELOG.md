# ShikshaSetu Changelog

## [2.0.0] - December 2025

### 🆕 Added
- **Q&A System with RAG**: Intelligent document question answering using Retrieval-Augmented Generation
- **pgvector Integration**: Semantic search with 384-dimensional embeddings
- **Supabase Migration**: Moved from local PostgreSQL to Supabase cloud database
- **3 New Q&A Endpoints**: `/api/v1/qa/process`, `/api/v1/qa/ask`, `/api/v1/qa/history/{id}`
- **RAG Service**: New `src/services/rag_service.py` with chunking, embedding, and vector search
- **Q&A Tasks**: 3 new Celery tasks in `src/tasks/qa_tasks.py`
- **Auto Q&A Processing**: Upload endpoint now automatically processes documents for Q&A
- **Chat History**: Complete Q&A conversation tracking with context

### 🔧 Changed
- **Database**: Migrated from PostgreSQL 16 to Supabase PostgreSQL 17.6
- **Models**: Consolidated from `src/repository/` into `src/models.py` and `src/database.py`
- **Endpoints**: Increased from 18 to 21 total API endpoints
- **Celery Workers**: Increased concurrency from 2 to 3 workers
- **Documentation**: Comprehensive updates to README.md and docs/API.md

### 🗑️ Removed
- **Deprecated Code**: Removed entire `src/repository/` folder (7 files)
- **Old Documentation**: Deleted 8 redundant .md files:
  - `BACKEND_TEST_RESULTS.md`
  - `BACKEND_FIX_REPORT.md`
  - `CLEANUP_REPORT.md`
  - `SIMPLIFICATION_ANALYSIS.md`
  - `SIMPLIFICATION_IMPLEMENTATION.md`
  - `PROJECT_STRUCTURE.md`
  - `QA_IMPLEMENTATION_PLAN.md`
  - `SUPABASE_SETUP.md`
- **Redundant Scripts**: Deleted 11 shell scripts:
  - `check_backend_status.sh`
  - `test_backend.sh`
  - `quick_test.sh`
  - `shiksha_cli.sh`
  - `start_services.sh` (root duplicate)
  - `stop_services.sh` (root duplicate)
  - `start_system.sh`
  - `scripts/quick_setup.sh`
  - `scripts/start.sh`
  - `scripts/utils/start_services.sh`
  - `scripts/utils/stop_services.sh`
  - `scripts/utils/start_backend.sh`
- **Test Artifacts**: Deleted `full_test_results.txt`, `test_results.txt`, `test_pdf.txt`

### 🔄 Fixed
- **Import Errors**: Updated 15+ files with correct import paths after repository removal
- **SQLAlchemy Conflict**: Renamed `metadata` to `chunk_metadata` in DocumentChunk model
- **Supabase Connection**: Corrected pooler address format for Supabase connections

### 📊 Database Schema
New tables added:
- `document_chunks`: Stores text chunks with metadata
- `embeddings`: Stores 384-dim vectors with pgvector type
- `chat_history`: Stores Q&A conversations with context

### 📦 Dependencies Added
- `pymupdf==1.26.6` - PDF text extraction
- `chromadb==1.3.4` - Vector database utilities
- `langchain==0.4.1` - RAG framework
- `tiktoken==0.12.0` - Token counting
- `supabase==2.24.0` - Supabase Python client
- `sentence-transformers` - Embedding generation

### 📝 Documentation
- **README.md**: Completely updated with Q&A features, Supabase setup, new architecture
- **docs/API.md**: Added 3 Q&A endpoint documentation sections
- **IMPLEMENTATION_COMPLETE.md**: Condensed from 348 to 87 lines
- **CHANGELOG.md**: Created this changelog

### 🧪 Testing
- **test_qa_feature.sh**: New end-to-end Q&A testing script
- **test_all_features.sh**: Retained for comprehensive testing

### 📁 Project Organization
**Kept (Essential)**:
- `README.md` - Main documentation
- `IMPLEMENTATION_COMPLETE.md` - Q&A implementation summary
- `restart_backend.sh` - Quick restart utility
- `test_qa_feature.sh` - Q&A testing
- `test_all_features.sh` - Full feature testing

**Cleaned**: Removed 22 unnecessary files (8 docs + 11 scripts + 3 test artifacts)

---

## [1.0.0] - November 2025

### Initial Release
- Text simplification using Flan-T5
- Multi-language translation (IndicTrans2)
- NCERT curriculum validation
- Text-to-speech generation
- JWT authentication with rate limiting
- 18 API endpoints
- Celery async task processing
- Docker and Kubernetes support
- PostgreSQL 16 database
- Comprehensive test suite

---

**For detailed Q&A implementation, see**: `IMPLEMENTATION_COMPLETE.md`  
**For API documentation, see**: `docs/API.md`  
**For setup instructions, see**: `README.md`
