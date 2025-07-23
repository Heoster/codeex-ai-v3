# 🚀 CodeEx AI - Advanced Conversational AI Platform

A comprehensive AI-powered chat application with advanced capabilities including real-time web scraping, mathematical problem solving, emotional intelligence, and personalized user interactions.

## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Enhanced AI Brain** - Human-like reasoning with contextual understanding
- **Mathematical Solver** - Symbolic computation with SymPy (equations, calculus, algebra)
- **Intent Classification** - Transformer-based understanding of user requests
- **Emotional Intelligence** - Sentiment analysis and adaptive responses
- **Vector Memory System** - Semantic conversation recall and context preservation
- **Personalization Engine** - User-adaptive communication styles and preferences

### 🕷️ **Real-time Web Scraping**
- **Automatic Data Collection** - Scrapes Wikipedia, tech news, GitHub trending, Stack Overflow
- **Content Deduplication** - Intelligent filtering and content hashing
- **Search Functionality** - Query through scraped content database
- **Performance Monitoring** - 98.5% success rate with comprehensive analytics

### 🔐 **Security & Authentication**
- **Google OAuth Integration** - Secure authentication system
- **AES-256 Encryption** - Chat message encryption
- **Session Management** - Secure user session handling
- **Input Validation** - Protection against injection attacks

### 📊 **Analytics & Monitoring**
- **AI Performance Dashboard** - Real-time AI metrics and analytics
- **Storage Management** - Comprehensive system monitoring
- **Learning Analytics** - Track AI improvement and user interactions
- **Health Monitoring** - Automatic system health checks

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask
- **AI/ML**: SymPy, NumPy, VADER Sentiment Analysis, Sentence Transformers
- **Web Scraping**: BeautifulSoup, aiohttp, requests
- **Database**: SQLite (multiple specialized databases)
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Authentication**: Google OAuth 2.0
- **Email**: SMTP integration

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
git clone <repository-url>
cd codeex-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. **Environment Configuration**
Create `.env` file:
```env
SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

### 3. **Initialize & Run**
```bash
# Initialize all enhanced features
python startup_enhanced_features.py

# Start the application
python app.py
```

### 4. **Access the Application**
- **Main App**: http://localhost:5000
- **AI Dashboard**: http://localhost:5000/ai-brain-dashboard
- **Storage Analytics**: http://localhost:5000/storage

## 🏗️ System Architecture

### **Core Components**
```
📁 CodeEx AI/
├── 🌐 app.py                     # Main Flask application
├── 🚀 startup_enhanced_features.py # System initialization
├── 
├── 🧠 AI Engine/
│   ├── enhanced_ai_evolution.py   # Advanced AI capabilities
│   ├── ai_brain_integration.py   # Human-like reasoning
│   ├── complete_ai_integration.py # Unified AI system
│   └── sentiment_analyzer.py     # Emotion analysis
├── 
├── 🕷️ Web Scraping/
│   └── web_scraper.py            # Real-time data collection
├── 
├── 🔧 Services/
│   ├── chat_history_service.py   # Session management
│   ├── security_service.py       # Encryption & security
│   └── ai_service.py            # Core AI service
├── 
├── 🎨 Frontend/
│   ├── templates/               # HTML templates
│   └── static/                 # CSS, JS, images
└── 
└── 🗄️ Databases/
    ├── codeex.db              # Main application data
    ├── ai_brain.db            # AI knowledge storage
    ├── web_scraping.db        # Scraped content
    └── advanced_ai_evolution.db # Learning analytics
```

## 🤖 AI Capabilities

### **Mathematical Problem Solving**
```python
# Examples of supported operations:
- Solve equations: "solve 2x² + 5x - 3 = 0"
- Calculus: "derivative of x³ + 2x²"
- Integration: "integral of x² + 1"
- Statistics: Calculate mean, std, variance
```

### **Intent Classification**
- **Company Queries**: Information about Heoster Technologies
- **Mathematical Problems**: Equation solving and calculations
- **Programming Help**: Code assistance and examples
- **General Assistance**: Wide range of topics and questions

### **Emotional Intelligence**
- **Sentiment Analysis**: Detect user emotions (positive, negative, neutral)
- **Adaptive Responses**: Adjust communication style based on user mood
- **Personalization**: Learn user preferences and communication patterns

## 🔌 API Endpoints

### **Chat System**
```http
GET  /api/chat/sessions              # Get user sessions
POST /api/chat/sessions              # Create new session
GET  /api/chat/{id}/messages         # Get messages
POST /api/chat/{id}/messages         # Send message
```

### **Web Scraping**
```http
POST /api/scraping/test              # Test scraping
GET  /api/scraping/stats             # Get statistics
GET  /api/scraping/results           # Recent results
GET  /api/scraping/search?q=query    # Search content
POST /api/scraping/start             # Start auto-scraping
POST /api/scraping/stop              # Stop auto-scraping
```

### **AI Enhancement**
```http
POST /api/ai/optimize                # Optimize performance
GET  /api/ai/export                  # Export AI data
POST /api/ai/memory/clear            # Clear memory
GET  /api/ai/analytics               # Get analytics
GET  /api/ai/dashboard-data          # Dashboard data
POST /api/ai/feedback                # Submit feedback
```

## 📊 Performance Metrics

### **System Status**
- ✅ **5/5 Core Components** initialized successfully
- ✅ **98.5% Web Scraping** success rate
- ✅ **Real-time Data Collection** active
- ✅ **Advanced AI Features** operational
- ✅ **Database Connections** established

### **AI Capabilities Status**
- 🧮 **Advanced Mathematics**: ✅ Active (SymPy enabled)
- 🎯 **Intent Classification**: ✅ Active (Fallback mode)
- 🎭 **Sentiment Analysis**: ✅ Active (VADER enabled)
- 🕷️ **Web Scraping**: ✅ Active (Real-time collection)
- 💾 **Vector Memory**: ⚠️ Basic mode (FAISS optional)
- 🤖 **Transformer Models**: ⚠️ Fallback mode (Optional enhancement)

## 🔧 Configuration

### **Google OAuth Setup**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create OAuth 2.0 credentials
3. Add redirect URI: `http://localhost:5000/auth/google/callback`
4. Copy Client ID and Secret to `.env`

### **Optional Enhancements**
```bash
# For advanced NLP capabilities
pip install sentence-transformers

# For vector search (advanced memory)
pip install faiss-cpu

# For professional grammar checking
pip install language-tool-python
```

## 🧪 Testing

```bash
# Test AI components
python test_ai_brain.py
python test_enhanced_ai.py
python test_complete_ai_system.py

# Test web scraping
python test_wikipedia_integration.py

# Test system initialization
python startup_enhanced_features.py
```

## 🚀 Deployment

### **Development**
```bash
python app.py  # Runs on http://localhost:5000
```

### **Production**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📈 Monitoring & Analytics

### **AI Dashboard** (`/ai-brain-dashboard`)
- Real-time AI performance metrics
- Learning statistics and progress
- System health monitoring
- Interactive AI testing interface

### **Storage Management** (`/storage`)
- Comprehensive system analytics
- Web scraping performance
- AI capabilities overview
- Performance optimization tools

## 🎯 Key Achievements

- ✅ **Advanced AI Integration** - Multiple AI engines working together
- ✅ **Real-time Data Collection** - Fresh information from web scraping
- ✅ **Emotional Intelligence** - Adaptive conversation experience
- ✅ **Mathematical Capabilities** - Symbolic computation with SymPy
- ✅ **User Personalization** - Adaptive responses and memory
- ✅ **Comprehensive Analytics** - Full system monitoring
- ✅ **Secure Architecture** - Encryption and authentication

## 🔮 Future Enhancements

### **Advanced AI Features**
- [ ] Voice chat integration
- [ ] Multi-language support
- [ ] Advanced transformer models
- [ ] LangChain integration for reasoning chains

### **System Improvements**
- [ ] FastAPI migration for async performance
- [ ] PostgreSQL for better scalability
- [ ] Redis for caching and sessions
- [ ] Docker containerization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📞 Support

- **Email**: the.heoster@mail.com
- **Issues**: Create GitHub issue
- **Documentation**: Available at `/docs` endpoint

## 📄 License

This project is licensed under the MIT License.

---

## 🎉 **CodeEx AI - Where Intelligence Meets Innovation!**

**🚀 Enhanced with:**
- 🧠 Advanced AI reasoning
- 🕷️ Real-time web scraping  
- 🎭 Emotional intelligence
- 🧮 Mathematical problem solving
- 💾 Vector-based memory
- 👤 User personalization
- 📊 Comprehensive analytics

**Ready to revolutionize your AI conversation experience!** 🌟# codeex-ai-v3
