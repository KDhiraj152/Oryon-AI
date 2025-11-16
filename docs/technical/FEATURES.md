# 🎯 ShikshaSetu - Complete Feature List

**Production Backend Features for Frontend Implementation**

---

## 🔐 Authentication & User Management

### ✅ User Registration
- **Endpoint:** `POST /api/v1/auth/register`
- **Rate Limit:** 3 requests/hour
- **Features:**
  - Email validation (unique check)
  - Password hashing (bcrypt)
  - Minimum 8 characters password requirement
  - Full name field (optional)
  - Automatic JWT token generation on signup
  - Returns both access & refresh tokens
- **Frontend Needs:**
  - Registration form (email, password, full name)
  - Email format validation
  - Password strength indicator
  - Error handling for duplicate emails
  - Rate limit warning (show "3 attempts remaining")

### ✅ User Login
- **Endpoint:** `POST /api/v1/auth/login`
- **Rate Limit:** 5 requests/hour
- **Features:**
  - Email/password authentication
  - bcrypt password verification
  - Returns access token (30-min expiry)
  - Returns refresh token (7-day expiry)
  - Token stored in response
- **Frontend Needs:**
  - Login form (email, password)
  - "Remember me" checkbox
  - Token storage (localStorage/sessionStorage)
  - Auto-redirect after login
  - "Forgot password" link placeholder

### ✅ Token Refresh
- **Endpoint:** `POST /api/v1/auth/refresh`
- **Rate Limit:** 10 requests/hour
- **Features:**
  - Validates refresh token
  - Issues new access token
  - Extends session without re-login
  - Automatic token rotation
- **Frontend Needs:**
  - Silent token refresh (auto-trigger before expiry)
  - Axios interceptor for 401 errors
  - Refresh token storage
  - Logout on refresh failure

### ✅ Get Current User Profile
- **Endpoint:** `GET /api/v1/auth/me`
- **Features:**
  - Returns user ID, email, full name
  - Returns account status (active/inactive)
  - Shows account creation date
  - Protected endpoint (requires auth)
- **Frontend Needs:**
  - User profile page
  - Display user info in navbar/sidebar
  - Avatar/profile picture placeholder
  - Account settings page

---

## 📤 File Upload & Management

### ✅ Single File Upload
- **Endpoint:** `POST /api/v1/upload`
- **Rate Limit:** 10 requests/minute
- **Features:**
  - Max file size: 100MB
  - Supported formats: PDF, PNG, JPEG, JPG
  - Magic byte validation (prevents MIME spoofing)
  - Automatic file organization by date (YYYY/MM/DD)
  - Unique filename generation
  - Returns file path for processing
  - Progress tracking support
- **Frontend Needs:**
  - Drag-and-drop upload area
  - File picker button
  - Upload progress bar (0-100%)
  - File size validation (show size in MB)
  - File type validation before upload
  - Preview for images
  - "Cancel upload" button
  - Success confirmation with file name

### ✅ Chunked File Upload
- **Endpoint:** `POST /api/v1/upload/chunked`
- **Rate Limit:** 10 requests/minute
- **Features:**
  - Split large files into 5MB chunks
  - Resume-able uploads
  - Better for slow connections
  - Automatic chunk assembly
  - Progress tracking per chunk
  - Unique file ID generation
- **Frontend Needs:**
  - Chunked upload for files >20MB
  - Chunk progress indicator (3/10 chunks)
  - Resume upload on failure
  - Network status detection
  - "Uploading chunk X of Y" message

---

## 🤖 AI Content Processing

### ✅ Full AI Pipeline
- **Endpoint:** `POST /api/v1/process`
- **Rate Limit:** 5 requests/minute
- **Features:**
  - **Stage 1: OCR Text Extraction**
    - PDF text extraction (PyMuPDF)
    - Image OCR (Tesseract)
    - Math formula preservation
    - Multi-language support (10 Indian languages + English)
  - **Stage 2: Text Simplification**
    - Flan-T5 model
    - Grade-level adaptation (5-12)
    - Subject-specific simplification
    - Preserves key concepts
  - **Stage 3: Translation**
    - IndicTrans2 model
    - 10 Indian languages supported
    - Multiple target languages per request
    - Maintains educational context
  - **Stage 4: Content Validation**
    - BERT semantic similarity
    - NCERT standards alignment
    - Quality score (0-1)
    - Confidence metrics
  - **Stage 5: Audio Generation**
    - VITS + Coqui TTS
    - Multilingual audio synthesis
    - Natural voice quality
    - MP3 format output
- **Input Parameters:**
  - `file_path` (from upload)
  - `grade_level` (5-12)
  - `subject` (Science, Math, History, etc.)
  - `target_languages` (array: Hindi, Tamil, etc.)
  - `output_format` (text/audio/both)
  - `validation_threshold` (0.0-1.0)
- **Frontend Needs:**
  - Multi-step processing wizard
  - Grade level selector (5-12)
  - Subject dropdown (6 subjects)
  - Language multi-select (10 languages)
  - Output format toggle (text/audio/both)
  - Processing stage indicator (5 stages)
  - Real-time progress bar with stage names
  - Estimated time remaining
  - "Cancel processing" button
  - Results preview page

### ✅ Individual Processing Tasks

#### Text Simplification Only
- **Endpoint:** `POST /api/v1/simplify`
- **Features:**
  - Input: raw text + grade level + subject
  - Output: simplified text
  - Faster than full pipeline
  - No rate limit
- **Frontend Needs:**
  - Text input area
  - Grade selector
  - Subject selector
  - "Simplify" button
  - Side-by-side comparison (before/after)

#### Translation Only
- **Endpoint:** `POST /api/v1/translate`
- **Features:**
  - Input: text + target languages + grade
  - Output: translated versions
  - Multiple languages at once
- **Frontend Needs:**
  - Text input
  - Language checkboxes (multi-select)
  - "Translate" button
  - Tabbed output (one tab per language)

#### Validation Only
- **Endpoint:** `POST /api/v1/validate`
- **Features:**
  - Input: original + simplified text + grade + subject
  - Output: validation score + alignment metrics
  - NCERT standards checking
- **Frontend Needs:**
  - Two text inputs (original & simplified)
  - Grade/subject selectors
  - "Validate" button
  - Score display (percentage)
  - Quality indicators (accuracy, alignment)

#### Audio Generation Only
- **Endpoint:** `POST /api/v1/tts`
- **Features:**
  - Input: text + language
  - Output: MP3 audio file
  - Natural voice synthesis
- **Frontend Needs:**
  - Text input area
  - Language selector
  - "Generate Audio" button
  - Audio player with controls
  - Download button

---

## 📊 Task Management & Monitoring

### ✅ Task Status Polling
- **Endpoint:** `GET /api/v1/tasks/{task_id}`
- **Features:**
  - Real-time task progress (0-100%)
  - Current processing stage
  - Status messages
  - Task states: PENDING, STARTED, PROCESSING, SUCCESS, FAILURE
  - Estimated time remaining
  - Error details on failure
  - Full result on success
- **Frontend Needs:**
  - Auto-polling (every 3 seconds)
  - Progress bar with percentage
  - Stage indicator (visual steps)
  - Status messages display
  - Error dialog on failure
  - Success notification
  - Stop polling when complete

### ✅ Task Cancellation
- **Endpoint:** `DELETE /api/v1/tasks/{task_id}`
- **Features:**
  - Graceful task termination
  - Force kill option (terminate=true)
  - Immediate response
  - Cleanup of partial results
- **Frontend Needs:**
  - "Cancel" button during processing
  - Confirmation dialog ("Are you sure?")
  - Force cancel option for stuck tasks
  - Cancellation feedback message

---

## 📥 Content Retrieval & Management

### ✅ Get Processed Content
- **Endpoint:** `GET /api/v1/content/{content_id}`
- **Features:**
  - Returns complete processed content
  - Original text included
  - Simplified text
  - All translations
  - Validation scores
  - Audio availability status
  - Metadata (grade, subject, date)
  - User ownership verification
- **Frontend Needs:**
  - Content detail page
  - Tabbed interface (original/simplified/translations)
  - Copy-to-clipboard buttons
  - Print-friendly view
  - Download as PDF option
  - Share link generation
  - Edit metadata button

### ✅ Stream Audio Files
- **Endpoint:** `GET /api/v1/audio/{content_id}`
- **Query Params:** `?language=Hindi`
- **Features:**
  - MP3 audio streaming
  - Language-specific audio
  - Content-Disposition header for download
  - Support for audio players
  - Proper MIME type
- **Frontend Needs:**
  - HTML5 audio player with controls
  - Play/pause/seek
  - Volume control
  - Download audio button
  - Multiple audio players (one per language)
  - Playlist support for multiple contents

### ✅ Submit Feedback
- **Endpoint:** `POST /api/v1/feedback`
- **Features:**
  - Rating system (1-5 stars)
  - Text comment (up to 1000 chars)
  - Feedback types: simplification, translation, audio, validation, general
  - Linked to specific content
  - User attribution
  - Timestamp tracking
- **Frontend Needs:**
  - Star rating component (1-5)
  - Text area for comments
  - Feedback type radio buttons
  - "Submit Feedback" button
  - Thank you message
  - Feedback history page

---

## 🏥 System Health & Monitoring

### ✅ Basic Health Check
- **Endpoint:** `GET /health`
- **Features:**
  - Quick status check
  - Response time < 100ms
  - Returns "healthy" or "unhealthy"
  - Timestamp included
- **Frontend Needs:**
  - Status indicator in footer (green dot)
  - Tooltip on hover showing last check
  - Auto-refresh every 60 seconds
  - Warning banner if unhealthy

### ✅ Detailed Health Check
- **Endpoint:** `GET /health/detailed`
- **Features:**
  - **Database Status:**
    - Connection health
    - Latency in milliseconds
    - Error details
  - **Redis Status:**
    - Cache availability
    - Connection latency
  - **Celery Status:**
    - Active workers count
    - Active tasks count
    - Queue depth
  - **Storage Status:**
    - Disk usage percentage
    - Free space in GB
    - Upload directory status
  - **System Metrics:**
    - CPU usage percentage
    - Memory usage
    - Available memory
  - Overall health verdict
  - Check duration
- **Frontend Needs:**
  - System status dashboard
  - Health check cards (one per component)
  - Color-coded status (green/yellow/red)
  - Metrics charts (CPU, memory, disk)
  - Refresh button
  - Last updated timestamp
  - Admin-only access

---

## 🎨 Supported Languages

### ✅ Translation & Audio Languages (10)
1. **Hindi** (हिंदी) - Most popular
2. **Tamil** (தமிழ்)
3. **Telugu** (తెలుగు)
4. **Bengali** (বাংলা)
5. **Marathi** (मराठी)
6. **Gujarati** (ગુજરાતી)
7. **Kannada** (ಕನ್ನಡ)
8. **Malayalam** (മലയാളം)
9. **Punjabi** (ਪੰਜਾਬੀ)
10. **Odia** (ଓଡ଼ିଆ)

**Frontend Implementation:**
- Language selector with native script names
- Flag icons for visual identification
- Multi-select for translations
- Language tabs in results view
- Audio player for each language

---

## 📚 Supported Subjects (6)

1. **Science** - Biology, Physics, Chemistry
2. **Mathematics** - Algebra, Geometry, Arithmetic
3. **History** - Indian & World History
4. **Geography** - Physical & Human Geography
5. **English** - Language & Literature
6. **Social Studies** - Civics, Economics

**Frontend Implementation:**
- Subject dropdown with icons
- Subject-specific color coding
- Subject filter in content library
- Subject-based recommendations

---

## 🎓 Grade Levels (8)

**Supported:** Classes 5-12

**Frontend Implementation:**
- Grade selector (dropdown or slider)
- Grade-appropriate UI complexity
- Age-appropriate color schemes
- Difficulty indicators

---

## 🔒 Security Features

### ✅ Rate Limiting
- **Upload:** 10 requests/minute
- **Process:** 5 requests/minute
- **Register:** 3 requests/hour
- **Login:** 5 requests/hour
- **Refresh:** 10 requests/hour
- Rate limit headers in responses
- Retry-After header when exceeded

**Frontend Implementation:**
- Rate limit warnings before hitting limit
- Countdown timer when limited
- "X attempts remaining" indicator
- Disable button when rate limited
- Queue requests if needed

### ✅ JWT Authentication
- Access token: 30-minute expiry
- Refresh token: 7-day expiry
- Automatic token refresh
- Secure token storage
- Bearer token in headers

**Frontend Implementation:**
- Token management service
- Axios interceptors
- Automatic refresh on 401
- Logout on refresh failure
- Session timeout warnings

### ✅ Input Validation
- File type validation (magic bytes)
- File size limits (100MB)
- Email format validation
- Password strength requirements
- Text length limits
- SQL injection prevention
- XSS protection

**Frontend Implementation:**
- Client-side validation before submission
- Real-time validation feedback
- Error messages for invalid inputs
- Sanitize user inputs
- Preview before submission

---

## 📱 Progressive Features for Frontend

### File Upload Enhancements
- [ ] Multiple file upload (batch processing)
- [ ] Upload history/queue
- [ ] Pause/resume uploads
- [ ] Upload from URL
- [ ] Drag-and-drop reordering
- [ ] File preview before upload

### Content Library
- [ ] Search processed content
- [ ] Filter by language/grade/subject
- [ ] Sort by date/rating
- [ ] Bookmark favorites
- [ ] Share content links
- [ ] Export content (PDF/JSON)
- [ ] Print optimized view

### User Dashboard
- [ ] Processing history
- [ ] Usage statistics
- [ ] Monthly processing quota
- [ ] Recent uploads
- [ ] Favorite content
- [ ] Account settings

### Collaboration Features
- [ ] Share content with others
- [ ] Public/private content toggle
- [ ] Content collections/folders
- [ ] Collaborative editing (future)
- [ ] Comment system (future)

### Offline Support
- [ ] Cache processed content
- [ ] Offline audio playback
- [ ] Service worker for PWA
- [ ] Sync when back online
- [ ] Download for offline use

### Accessibility
- [ ] Screen reader support
- [ ] Keyboard navigation
- [ ] High contrast mode
- [ ] Font size adjustment
- [ ] Color blind friendly
- [ ] ARIA labels

### Mobile Optimization
- [ ] Responsive design
- [ ] Touch gestures
- [ ] Mobile-first UI
- [ ] Bottom navigation
- [ ] Swipe between languages
- [ ] Camera upload (mobile)

### Analytics & Insights
- [ ] Processing time stats
- [ ] Most used languages
- [ ] Subject distribution
- [ ] Grade level usage
- [ ] Quality metrics over time
- [ ] User engagement metrics

---

## 🎨 Recommended Frontend Architecture

### Pages/Routes
```
/login                  - Login page
/register              - Registration page
/dashboard             - User dashboard (after login)
/upload                - File upload page
/process/:id           - Processing status page
/content/:id           - Content detail/result page
/library               - Content library/history
/profile               - User profile & settings
/health                - System health (admin)
```

### Key Components
```
<AuthProvider>         - Auth context wrapper
<FileUpload>           - Drag-drop upload component
<ProgressTracker>      - Task progress with stages
<ContentViewer>        - Tabbed content display
<AudioPlayer>          - Multi-language audio player
<LanguageSelector>     - Multi-select languages
<GradeSelector>        - Grade level picker
<SubjectSelector>      - Subject dropdown
<RatingStars>          - 5-star rating component
<HealthCard>           - System health indicator
<RateLimitWarning>     - Rate limit notification
```

### State Management Recommendations
- **Auth State:** User, tokens, session
- **Upload State:** Files, progress, status
- **Task State:** Task ID, progress, stage, result
- **Content State:** Processed content, audio URLs
- **UI State:** Loading, errors, notifications

### API Service Layer
```javascript
authService           - register, login, refresh, getUser
uploadService         - uploadFile, uploadChunked
processService        - process, simplify, translate, validate, tts
taskService           - getStatus, cancelTask, pollTask
contentService        - getContent, getAudio, submitFeedback
healthService         - checkHealth, getDetailedHealth
```

---

## 🚀 Quick Start for Frontend Developers

### 1. Authentication Flow
```javascript
// Register → Login → Get user → Store tokens
register() → login() → getCurrentUser() → localStorage.setItem()

// On every request
headers: { Authorization: `Bearer ${accessToken}` }

// Handle token expiry
401 → refresh() → retry request → or logout()
```

### 2. Content Processing Flow
```javascript
// Upload → Process → Poll → Display results
uploadFile() 
  → process({ file_path, grade, subject, languages }) 
  → pollTaskStatus(task_id) 
  → getContent(content_id)
  → display results
```

### 3. Error Handling
```javascript
// All API errors return structured format
{
  "error": "ERROR_CODE",
  "message": "User-friendly message",
  "timestamp": "2025-11-16T..."
}

// Handle common errors
400 → Show validation errors
401 → Redirect to login
429 → Show rate limit message with retry time
500 → Show generic error, log to console
```

---

## 📊 Performance Metrics

### API Response Times
- Health check: <100ms
- Login/Register: <500ms
- File upload: Depends on size (show progress)
- Task submission: <200ms
- Task status: <100ms
- Content retrieval: <300ms

### Processing Times (Typical)
- OCR extraction: 15-30 seconds
- Simplification: 20-40 seconds
- Translation (per language): 30-60 seconds
- Validation: 10-20 seconds
- Audio generation (per language): 20-40 seconds

**Full pipeline:** 2-5 minutes depending on file size and number of languages

### Frontend Performance Goals
- First Contentful Paint: <1.5s
- Time to Interactive: <3s
- Lighthouse Score: >90
- Bundle size: <500KB (gzipped)

---

## 🎯 MVP Features for First Release

### Must-Have (P0)
✅ User registration & login  
✅ File upload with progress  
✅ Full AI pipeline processing  
✅ Task progress tracking  
✅ Content viewing (all languages)  
✅ Audio playback  
✅ Basic error handling  

### Should-Have (P1)
- Content library/history
- User profile page
- Feedback submission
- Download results
- Mobile responsive

### Nice-to-Have (P2)
- Advanced search/filter
- Bookmarks/favorites
- Share functionality
- Offline support
- Analytics dashboard

---

**Ready for frontend development! 🚀**

*Last Updated: 16 November 2025*
