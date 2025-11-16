import { Link } from 'react-router-dom';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
        <div className="text-center">
          <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600 bg-clip-text text-transparent animate-gradient">
            ShikshaSetu
          </h1>
          <p className="text-2xl md:text-3xl text-gray-800 font-semibold mb-4">
            AI-Powered Multilingual Education
          </p>
          <p className="text-lg text-gray-600 mb-10 max-w-2xl mx-auto">
            Transform educational content into multiple Indian languages with AI simplification, 
            translation, and text-to-speech capabilities.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link 
              to="/login" 
              className="btn-primary px-8 py-4 text-lg font-semibold transform hover:scale-105 transition-all shadow-lg"
            >
              Get Started Free →
            </Link>
            <Link 
              to="/about" 
              className="btn-secondary px-8 py-4 text-lg font-semibold transform hover:scale-105 transition-all"
            >
              Learn More
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="glass-card p-6 text-center hover:scale-105 transition-transform">
            <div className="text-5xl mb-4">🌍</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">10 Languages</h3>
            <p className="text-gray-600">Support for Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and Odia</p>
          </div>
          
          <div className="glass-card p-6 text-center hover:scale-105 transition-transform">
            <div className="text-5xl mb-4">🤖</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">AI Simplification</h3>
            <p className="text-gray-600">Intelligent text simplification adapted to grade levels 5-12 for better comprehension</p>
          </div>
          
          <div className="glass-card p-6 text-center hover:scale-105 transition-transform">
            <div className="text-5xl mb-4">🎧</div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">Audio Support</h3>
            <p className="text-gray-600">High-quality text-to-speech conversion for accessible learning experiences</p>
          </div>
        </div>

        {/* Stats Section */}
        <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="p-4">
            <div className="text-4xl font-bold text-primary-600">10+</div>
            <div className="text-gray-600 mt-1">Languages</div>
          </div>
          <div className="p-4">
            <div className="text-4xl font-bold text-primary-600">6</div>
            <div className="text-gray-600 mt-1">Subjects</div>
          </div>
          <div className="p-4">
            <div className="text-4xl font-bold text-primary-600">5-12</div>
            <div className="text-gray-600 mt-1">Grade Levels</div>
          </div>
          <div className="p-4">
            <div className="text-4xl font-bold text-primary-600">AI</div>
            <div className="text-gray-600 mt-1">Powered</div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="mt-20 glass-card p-10 text-center">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            Ready to Transform Education?
          </h2>
          <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
            Join educators and students across India in making quality education accessible in every language.
          </p>
          <Link 
            to="/register" 
            className="btn-primary px-10 py-4 text-lg font-semibold inline-block transform hover:scale-105 transition-all shadow-lg"
          >
            Create Free Account
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
